# Functional Diversity Analysis

This project compares the electrostatic fields of molecules to identify structurally unique regions. It uses 3D field calculations (via Psi4), field alignment, and difference analysis to map molecular uniqueness back to functional motifs, producing a table of unique SMILES and visualisations of the unique regions for each molecule.

## Key Features
- 3D geometry generation with RDKit
- SCF electrostatic field calculation with Psi4
- List-to-list field alignment and overlap comparison
- Unique field region mapping to atoms
- SMILES generation and visualisation of unique motifs
- Results saved as annotated CSV and image files

## Project Structure
```
functional-diversity/
├── scripts/
│ └── functional_diversity.py
├── Dockerfile
├── README.md
├── .gitignore
└── outputs/
```

## Dependencies
This project was run in a containerized environment using Docker. A pre-built `Dockerfile` is included to simplify installation of Psi4 and other dependencies.

## License
MIT License
