# Plasmodium Sporozoite Infection Simulation

This project simulates the early stages of Plasmodium sporozoite infection, modeling the journey from mosquito salivary glands through dermal tissue to blood vessels.

## Project Structure

```
SporozoiteInfect/
├── main_simulation.py          # Main simulation script
├── utils/
│   └── sporozoite.py          # Base Sporozoite class
├── stages/
│   ├── dermal/
│   │   └── dermal_stage.py    # Dermal stage simulation
│   └── vascular/
│       └── vascular_stage.py  # Vascular stage simulation
└── README.md
```

## Features

### Sporozoite Properties
- **Curved deformable rods**: Modeled with length, width, and curvature parameters
- **Motility**: Individual movement capabilities with random walk behavior
- **Deformation**: Response to tissue pressure and environmental forces
- **Viability**: Decreases due to immune responses and environmental stress

### Dermal Stage
- **Salivary Gland**: Sporozoite reservoir with controlled extrusion
- **Tissue Resistance**: Collagen fiber networks impeding movement
- **Immune Response**: Simulated immune cell interactions
- **Environmental Forces**: Tissue pressure causing deformation

### Vascular Stage
- **Blood Vessel Network**: Realistic vascular topology with vessels of varying sizes
- **Chemotaxis**: Attraction-based migration toward blood vessels
- **Vessel Penetration**: Probability-based entry into circulation
- **Blood Flow**: Movement within vessels following flow dynamics

## Installation & Requirements

```bash
pip install numpy matplotlib
```

## Usage

### Basic Simulation
```bash
python main_simulation.py
```

### Custom Parameters
```bash
# Run with 50 sporozoites for 100 time units
python main_simulation.py --sporozoites 50 --time 100

# Run without visualization (headless mode)
python main_simulation.py --no-viz
```

### Command Line Options
- `--sporozoites, -s`: Number of initial sporozoites (default: 30)
- `--time, -t`: Maximum simulation time (default: 50.0)
- `--no-viz`: Disable visualization for batch runs

## Biological Basis

The simulation is based on the following biological processes:

1. **Salivary Gland Extrusion**: Sporozoites are injected from mosquito salivary glands into host dermis during blood feeding
2. **Dermal Migration**: Sporozoites navigate through dermal tissue, encountering:
   - Collagen fiber resistance
   - Immune cell surveillance
   - Tissue pressure causing deformation
3. **Vascular Targeting**: Sporozoites exhibit chemotactic behavior toward blood vessels
4. **Vessel Entry**: Successful sporozoites penetrate vessel walls and enter circulation

## Visualization

The simulation provides real-time visualization with two panels:
- **Left Panel**: Dermal stage showing salivary gland (red circle) and sporozoite movement
- **Right Panel**: Vascular stage showing blood vessel network and migration patterns

Color coding:
- **Red**: Sporozoites in salivary gland
- **Orange**: Sporozoites in dermal tissue
- **Green**: Sporozoites migrating toward vessels
- **Blue**: Sporozoites in blood vessels

## Output Statistics

The simulation tracks and reports:
- Extrusion success rate
- Dermal survival rate
- Vessel entry success rate
- Overall infection success rate

## Scientific Applications

This simulation can be used to:
- Study sporozoite transmission dynamics
- Evaluate intervention strategies
- Understand tissue-pathogen interactions
- Model malaria infection probability

## License

This simulation is provided for educational and research purposes.