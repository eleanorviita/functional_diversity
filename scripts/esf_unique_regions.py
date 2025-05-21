'''
Functional Diversity Analysis Script

Compares two molecule lists (List A vs List B) to identify unique electrostatic field regions in List A molecules.
Outputs include a CSV unique SMILES summary and visual representations of distinct molecular substructures.

'''

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import rotate
import psi4
import matplotlib.pyplot as plt
import os
import pickle
import signal
import re


# --- Timeout handling ---

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# --- Molecule, MoleculeAligner, FieldCalculator, Workflow classes and functions ---

class Molecule:

    def __init__(self, smiles):
        self.original_smiles = smiles
        self.cleaned_smiles = self.get_largest_fragment()
        self.mol = Chem.MolFromSmiles(self.cleaned_smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Infer charge and multiplicity
        self.charge = self._infer_formal_charge()
        self.multiplicity = self._infer_multiplicity()

        self.geometry = None
        self.field = None
        self.failed = False

    def _infer_formal_charge(self):
        return Chem.GetFormalCharge(self.mol)

    def _infer_multiplicity(self):
        num_radicals = sum([atom.GetNumRadicalElectrons() for atom in self.mol.GetAtoms()])
        # Multiplicity is 2S+1, where S is total spin
        return num_radicals + 1 if num_radicals > 0 else 1


    def get_largest_fragment(self):
        largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()

        mol = Chem.MolFromSmiles(self.original_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.original_smiles}")

        largest_frag = largest_fragment_chooser.choose(mol)
        cleaned_smiles = Chem.MolToSmiles(largest_frag)

        return cleaned_smiles

    def generate_geometry(self):

        # Use LargestFragmentChooser to select the main fragment
        #lfc = rdMolStandardize.LargestFragmentChooser()
        #self.mol = lfc.choose(self.mol)

        self.mol = Chem.AddHs(self.mol)
        result = AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        if result != 0:
            print(f"Failed to generate 3D geometry for molecule {self.original_smiles}")
        AllChem.UFFOptimizeMolecule(self.mol)
        self.geometry = self._extract_xyz()


    def _extract_xyz(self):
        conf = self.mol.GetConformer()
        atoms = [
            f"{atom.GetSymbol()} {conf.GetAtomPosition(i).x:.6f} {conf.GetAtomPosition(i).y:.6f} {conf.GetAtomPosition(i).z:.6f}"
            for i, atom in enumerate(self.mol.GetAtoms())
        ]
        return f"{len(atoms)}\n\n" + "\n".join(atoms)


class FieldCalculator:
    def __init__(self, molecule):
        self.molecule = molecule
        if self.molecule.geometry is None:
            self.molecule.generate_geometry()

    def compute_field(self, grid_limits=(-20, 20), grid_size=50, timeout=86400):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            print(f"Starting field computation for {self.molecule.original_smiles} (Timeout: {timeout // 3600}h)")
            self.molecule.mol = Chem.AddHs(self.molecule.mol)

            atom_block = ""
            conf = self.molecule.mol.GetConformer()
            for atom in self.molecule.mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_block += f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"

            psi4_input = f"""{self.molecule.charge} {self.molecule.multiplicity}\n{atom_block}"""

            reference_method = "rhf" if self.molecule.multiplicity == 1 else "uhf"

            psi4.set_num_threads(8)
            psi4.set_memory('4 GB')
            psi4.core.set_output_file('psi4_output.dat', False)
            psi4.set_options({
                'reference': reference_method,
                'guess': 'sad',  # Use superposition of atomic densities for initial guess
                'basis': '3-21G',  # Simple basis for faster calcs. For higher precision (but slower) use: '6-31G*'
                'e_convergence': 1e-4,  # Looser energy convergence
                'd_convergence': 1e-4,  # Looser density convergence
                'scf_type': 'df'
            })

            try:
                mol = psi4.geometry(self.molecule.geometry)
                energy, wavefunction = psi4.energy('scf', return_wfn=True)

            except TimeoutException:
                print(f"Timeout reached for {self.molecule.original_smiles}. Skipping...")
                self.molecule.failed = True
                self.molecule.field = None
                return

            except Exception as e:
                print(
                    f"Psi4 ESF calculation failed for {self.molecule.original_smiles} using {reference_method}. Trying alternative method...")

                # **Fallback to another method if RHF/UHF fails**
                fallback_method = "uhf" if reference_method == "rhf" else "rhf"

                try:
                    psi4.set_options({'reference': fallback_method})
                    energy, wavefunction = psi4.energy('scf', return_wfn=True)
                    print(f"Successfully computed ESF using fallback method: {fallback_method}")

                except Exception as e:
                    print(f"Fallback method {fallback_method} also failed for {self.molecule.original_smiles}")
                    self.molecule.failed = True
                    self.molecule.field = None
                    return

        finally:
            signal.alarm(0)

        if not wavefunction:
            print(f"No valid wavefunction for {self.molecule.original_smiles}. Skipping ESF calculation.")
            self.molecule.failed = True
            return

        try:
            axis_range = np.linspace(grid_limits[0], grid_limits[1], grid_size)
            xv, yv, zv = np.meshgrid(axis_range, axis_range, axis_range, indexing='ij')
            grid_points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

            esp_properties = psi4.core.ESPPropCalc(wavefunction)
            esp_values = esp_properties.compute_esp_over_grid_in_memory(psi4.core.Matrix.from_array(grid_points))

            esp_values_np = np.array(esp_values)

            field_shape = (grid_size, grid_size, grid_size)
            self.molecule.field = esp_values_np.reshape(field_shape)

            print(f"Successfully computed ESF for {self.molecule.original_smiles}")

        except Exception as e:
            print(f"ESF computation failed for {self.molecule.original_smiles}: {e}")
            self.molecule.failed = True
            self.molecule.field = None


    def visualize_field(self, field=None, title=None, index=0):
        if field is None:
            field = self.molecule.field

        if field is not None:
            mid_slice = field.shape[2] // 2
            vmin = np.percentile(field, 5)  # 5th percentile
            vmax = np.percentile(field, 95)  # 95th percentile

            plt.figure(figsize=(6, 5))
            plt.imshow(field[:, :, mid_slice], cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(label="Electrostatic Potential")
            plt.title(f"ESP Field for {self.molecule.cleaned_smiles}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")

            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            if title:
                plot_path = os.path.join(output_dir, f"{title}.png")
            else:
                plot_path = os.path.join(output_dir, f"esp_field_{index}.png")
            plt.savefig(plot_path)

            plt.close()  # Close the figure
        else:
            print("No field data available to visualize.")

class MoleculeAligner:
    def __init__(self, reference_molecule):
        self.reference = reference_molecule
        if self.reference.field is None:
            FieldCalculator(self.reference).compute_field()

    def align_molecule(self, molecule):
        if molecule.failed:
            print(f"Skipping alignment: {molecule.original_smiles} failed field computation.")
            return None

        if self.reference.failed:
            print(
                f"Skipping alignment: Reference molecule {self.reference.original_smiles} failed field computation.")
            return None

        if molecule.field is None:
            FieldCalculator(molecule).compute_field()

        ref_field = self.reference.field
        comparison_field = molecule.field

        def overlap_loss(rotation_angles):
            rotated_field = self._rotate_field(comparison_field, rotation_angles)
            loss = -np.sum(ref_field * rotated_field)  # Negative overlap for minimization
            return loss

        # Optimize the rotational alignment
        result = minimize(overlap_loss, x0=(0, 0, 0), bounds=[(-np.pi, np.pi)] * 3)
        optimal_rotation = result.x

        aligned_field = self._rotate_field(comparison_field, optimal_rotation)

        molecule.field = aligned_field
        return aligned_field, optimal_rotation

    def _rotate_field(self, field, angles):
        # Convert angles from radians to degrees since `rotate` expects degrees
        angles_deg = np.degrees(angles)

        rotated_field = rotate(field, angle=angles_deg[0], axes=(1, 2), reshape=False, order=3)
        rotated_field = rotate(rotated_field, angle=angles_deg[1], axes=(0, 2), reshape=False, order=3)
        rotated_field = rotate(rotated_field, angle=angles_deg[2], axes=(0, 1), reshape=False, order=3)

        return rotated_field


class Workflow:
    def __init__(self, molecules=None, output_dir="outputs"):
        self.molecules = molecules if molecules is not None else []
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    def process_molecules(self, molecules):
        print(f"Processing {len(molecules)} molecules...")

        for mol in molecules:
            if mol.field is None:
                print(f"Processing molecule: {mol.original_smiles}")
                if mol.geometry is None:
                    mol.generate_geometry()
                field_calc = FieldCalculator(mol)
                field_calc.compute_field()

        self.molecules = [mol for mol in self.molecules if not getattr(mol, 'failed', False)]

        print("Finished processing molecules.")

    def compute_all_vs_all_overlap(self, threshold=0.1):
        print("Identifying unique regions for each molecule...")

        def compute_overlap(ref_molecule, cmp_molecule):
            #print(f"Comparing {cmp_molecule.original_smiles} to reference {ref_molecule.original_smiles}...")

            aligner = MoleculeAligner(ref_molecule)
            aligner.align_molecule(cmp_molecule)

            if cmp_molecule.field is None:
                print(f"Error: Alignment failed for {cmp_molecule.original_smiles}")
                return None

            return cmp_molecule.field

        results = {}

        for i, ref_molecule in enumerate(self.molecules):
            print(f"\n Processing reference molecule {ref_molecule.original_smiles}...")

            ref_field = ref_molecule.field
            if ref_field is None:
                print(f"Error: No field data for reference molecule {ref_molecule.original_smiles}")
                continue

            combined_overlap = np.zeros_like(ref_field)

            comparison_molecules = [cmp for j, cmp in enumerate(self.molecules) if i != j and cmp.field is not None]

            for cmp_molecule in comparison_molecules:
                aligned_field = compute_overlap(ref_molecule, cmp_molecule)

                if aligned_field is None:
                    continue

                combined_overlap = np.maximum(combined_overlap, aligned_field)

            unique_regions = ref_field - combined_overlap
            unique_regions[unique_regions < threshold] = 0
            results[ref_molecule.original_smiles] = unique_regions

            print(f"Unique regions calculated for {ref_molecule.original_smiles}")

        return results

    def map_unique_regions_to_atoms(self, unique_regions, reference_molecule, voxel_size=0.2):
        conf = reference_molecule.mol.GetConformer()

        atom_positions = np.array([
            (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
            for i in range(reference_molecule.mol.GetNumAtoms())
        ])

        voxel_indices = np.argwhere(unique_regions > 0)
        voxel_centers = voxel_indices * voxel_size

        atom_voxel_map = {i: 0 for i in range(len(atom_positions))}

        for voxel_center in voxel_centers:
            distances = np.linalg.norm(atom_positions - voxel_center, axis=1)
            nearest_atom_idx = int(np.argmin(distances))

            if nearest_atom_idx < len(atom_positions):
                atom_voxel_map[nearest_atom_idx] += 1

                bonded_atoms = [
                    bond.GetOtherAtomIdx(nearest_atom_idx)
                    for bond in reference_molecule.mol.GetAtomWithIdx(int(nearest_atom_idx)).GetBonds()
                ]
                for ba in bonded_atoms:
                    atom_voxel_map[ba] += 1

        return atom_voxel_map

    def generate_sub_smiles(self, atom_mapping, reference_molecule):
        unique_smiles = {}
        mol = reference_molecule.mol
        num_atoms = mol.GetNumAtoms()

        for atom_idx, count in atom_mapping.items():
            if count <= 0 or atom_idx < 0 or atom_idx >= num_atoms:
                continue

            try:
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = [
                    bond.GetOtherAtomIdx(atom_idx)
                    for bond in atom.GetBonds()
                ]
                atom_list = sorted(set([atom_idx] + neighbors))

                if any(idx >= num_atoms or idx < 0 for idx in atom_list):
                    print(f"Skipping invalid atom list: {atom_list}")
                    continue

                try:
                    fragment = Chem.PathToSubmol(mol, atom_list)
                    smiles = Chem.MolToSmiles(fragment)
                except Exception as e:
                    # Fallback to MolFragmentToSmiles if PathToSubMol fails
                    smiles = rdmolfiles.MolFragmentToSmiles(mol, atom_list)

                if smiles not in unique_smiles:
                    unique_smiles[smiles] = set()
                unique_smiles[smiles].update(atom_list)

            except Exception as e:
                print(f"Failed to extract SMILES for atom {atom_idx} in {reference_molecule.cleaned_smiles}: {e}")

        return {smiles: sorted(list(indices)) for smiles, indices in unique_smiles.items()}

    def save_results(self, all_results, output_file="unique_regions_results.csv"):
        results_df = pd.DataFrame(all_results)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, output_file)
        results_df.to_csv(output_path, index=False)
        print(f"All results saved to {output_path}")


    def run_and_save(self, list_a_file, list_b_file, threshold=0.1):
        if os.path.exists("list_a_safe.pkl") and os.path.exists("list_b.pkl"):
            print("Loading saved molecule fields...")
            with open("list_a_safe.pkl", "rb") as f:
                list_a = pickle.load(f)
            with open("list_b.pkl", "rb") as f:
                list_b = pickle.load(f)
        else:
            print("Importing molecules from CSV files...")
            list_a = Workflow.import_molecules(list_a_file)
            list_b = Workflow.import_molecules(list_b_file)

            print("Computing electrostatic fields...")
            self.process_molecules(list_a + list_b)

            self.save_molecules(list_a, "list_a.pkl")
            self.save_molecules(list_b, "list_b.pkl")

        # Remove failed molecules
        list_a = [mol for mol in list_a if not getattr(mol, "failed", False)]
        list_b = [mol for mol in list_b if not getattr(mol, "failed", False)]
        print(f"{len(list_a)} molecules in List A | {len(list_b)} in List B")

        all_results = []

        for mol_a in list_a:
            print(f"Comparing {mol_a.original_smiles} to all molecules in list B...")

            accumulated_field = np.copy(mol_a.field)
            aligner = MoleculeAligner(mol_a)

            for mol_b in list_b:
                aligned_field, best_angles = aligner.align_molecule(mol_b)
                if aligned_field is None:
                    continue

                accumulated_field -= aligned_field

            unique_region = np.where(accumulated_field >= threshold, accumulated_field, 0)

            atom_mapping = self.map_unique_regions_to_atoms(unique_region, mol_a)
            unique_smiles = self.generate_sub_smiles(atom_mapping, mol_a)

            result_dir = f"toxins_vs_metabolites_results/threshold_{threshold}"
            os.makedirs(result_dir, exist_ok=True)
            try:
                self.visualize_unique_atoms(mol_a, unique_smiles, output_dir=result_dir)
            except Exception as e:
                print(f"visualize_unique_atoms failed for {mol_a.original_smiles}. {e}")


            result_entry = {
                "Molecule_A": mol_a.original_smiles,
                "Unique_SMILES": ', '.join(unique_smiles),
                "Num_Unique_SMILES": len(unique_smiles)
            }
            all_results.append(result_entry)

        self.save_results(all_results, output_file=f"comparison_results_threshold_{threshold}.csv")
        print("Comparison complete. Results saved.")


    def import_molecules(file_path, smiles_column="SMILES"):
        encodings_to_try = ["utf-8", "ISO-8859-1", "utf-16", "utf-8-sig"]

        for encoding in encodings_to_try:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                data.columns = data.columns.str.strip()
                if smiles_column not in data.columns:
                    raise ValueError(
                        f"Column '{smiles_column}' not found in {file_path}. Available columns: {list(data.columns)}")

                return [Molecule(smiles) for smiles in data[smiles_column].dropna()]

            except UnicodeDecodeError:
                continue
            except ValueError as e:
                continue
            except Exception as e:
                print(f"Unexpected error with {encoding}: {e}")

        print(f"Failed to load {file_path} with available encodings.")
        return []

    def save_molecules(self, molecules, filename="molecules.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(molecules, f)
        print(f"Molecules saved to {filename}")

    def load_molecules(self, filename="molecules.pkl"):
        try:
            with open(filename, "rb") as f:
                molecules = pickle.load(f)
            print(f"Molecules loaded from {filename}")
            return molecules
        except FileNotFoundError:
            print(f"No saved molecule file found. Running new computation.")
            return []

    def visualize_unique_atoms(self, reference_molecule, unique_smiles_mapping, img_size=(500, 500), output_dir="toxins_vs_metabolites_results"):
        mol = reference_molecule.mol

        highlight_color = (1.0, 0.0, 0.0)
        highlight_atoms = set()

        for atom_indices in unique_smiles_mapping.values():
            if isinstance(atom_indices, int):
                highlight_atoms.add(atom_indices)
            elif isinstance(atom_indices, (list, tuple)):
                highlight_atoms.update(atom_indices)

        highlight_atoms = list(highlight_atoms)
        highlight_colors = {int(idx): highlight_color for idx in highlight_atoms}

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_colors
            )
            drawer.FinishDrawing()

            os.makedirs(output_dir, exist_ok=True)

            safe_smiles = re.sub(r'[\\/*?:"<>|]', "", reference_molecule.cleaned_smiles)
            output_path = os.path.join(output_dir, f"unique_region_{safe_smiles}.png")

            with open(output_path, "wb") as f:
                f.write(drawer.GetDrawingText())
            print(f"Visualisation saved: {output_path}")
        except Exception as e:
            print(f"Error visualising molecule {reference_molecule.original_smiles}")
            print(e)



# --- Determine grid size for visualisations ---

def get_molecule_bounds(molecule):
    conf = molecule.mol.GetConformer()
    atom_positions = np.array([
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(molecule.mol.GetNumAtoms())
    ])

    min_coords = atom_positions.min(axis=0)
    max_coords = atom_positions.max(axis=0)
    size = max_coords - min_coords  # Get bounding box size

    return min_coords, max_coords, size


def check_grid_size(molecule, grid_limits=(-20, 20), padding=2):
    min_coords, max_coords, size = get_molecule_bounds(molecule)

    # Compute grid span
    grid_span = grid_limits[1] - grid_limits[0] - 2 * padding  # Reduce padding
    molecule_max_size = max(size)

    if molecule_max_size > grid_span:
        print(f"Warning: Molecule {molecule.original_smiles} is too large for the grid!")
        print(f"   - Bounding box size: {size}")
        print(f"   - Max molecule size: {molecule_max_size:.2f} Å")
        print(f"   - Grid span: {grid_span:.2f} Å")
        return False
    else:
        return True


def check_all_molecules_grid_size(molecules, grid_limits=(-20, 20)):
    all_fit = True
    for mol in molecules:
        if not check_grid_size(mol, grid_limits):
            all_fit = False

    if not all_fit:
        print("Some molecules exceed the grid size! Consider increasing `grid_limits`.")
    else:
        print("All molecules fit within the grid.")


def check_molecule_list_from_csv(file_path, grid_limits=(-20, 20)):
    print(f"Loading molecules from {file_path} for grid size check...")

    # Step 1: Import molecules
    molecules = Workflow.import_molecules(file_path)

    # Step 2: Generate geometries for all molecules
    print("Generating 3D geometries for all molecules...")
    for molecule in molecules:
        try:
            molecule.generate_geometry()
        except Exception as e:
            print(f"Warning: Failed to generate geometry for {molecule.original_smiles}. Skipping...")

    # Step 3: Check if they fit within the grid
    print("Checking if molecules fit within the grid...")
    check_all_molecules_grid_size(molecules, grid_limits=grid_limits)

    print("Grid check complete.")

    return molecules  # Return processed molecules


# --- Remove problematic molecules ---


def remove_molecule_from_pickle(smiles_to_remove, input_path="list_a.pkl", output_path="list_a_safe.pkl"):
    try:
        with open(input_path, "rb") as f:
            molecules = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {input_path}: {e}")
        return

    cleaned_molecules = [mol for mol in molecules if mol.original_smiles != smiles_to_remove]

    removed_count = len(molecules) - len(cleaned_molecules)
    if removed_count == 0:
        print(f"No match found for SMILES: {smiles_to_remove}")
    else:
        print(f"Removed {removed_count} molecule(s) with SMILES: {smiles_to_remove}")

    try:
        with open(output_path, "wb") as f:
            pickle.dump(cleaned_molecules, f)
        print(f"Cleaned molecule list saved to {output_path}")
    except Exception as e:
        print(f"Failed to save cleaned list: {e}")


if __name__ == "__main__":

    # Input files contain a column of SMILES strings under the heading "SMILES"
    file_a = "toxins_test_list.csv"
    file_b = "metabolites_test_list.csv"
    thresholds = [0.04, 0.02, 0.01, 0.005] #0.1, 0.08, 0.06

    # Check grid size before running the workflow
    print("Checking molecule list A...")
    molecules_a = check_molecule_list_from_csv(file_a)

    print("Checking molecule list B...")
    molecules_b = check_molecule_list_from_csv(file_b)

    #remove_molecule_from_pickle("")


    workflow = Workflow(output_dir="toxins_vs_metabolites_results")

    for threshold in thresholds:
        workflow.run_and_save(file_a, file_b, threshold=threshold)
