#!/usr/bin/env python3
"""
Fixed PDB Interface Charge Analysis Tool
Clean version with proper function ordering
"""

# Install required packages if needed
import sys
import subprocess

def install_required_packages():
    """Install required packages if not available."""
    required_packages = ['biopython', 'pandas', 'numpy', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'biopython':
                import Bio
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            elif package == 'requests':
                import requests
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("🔧 Installing missing packages...")
        for package in missing_packages:
            print(f"   Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                sys.exit(1)
        return True
    return False

if install_required_packages():
    sys.exit(0)

# Now import all required modules
import os
import math
import time
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
    import numpy as np
    import requests
    from Bio import PDB
    from Bio.PDB import NeighborSearch, Selection
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Optional DSSP handling
try:
    from Bio.PDB.DSSP import DSSP
    has_dssp = True
except Exception:
    has_dssp = False

# Charged residue definitions with ACTUAL charge values
RESIDUE_CHARGES = {
    "ARG": +1.0,    # Arginine: fully protonated (pKa ~12.5)
    "LYS": +1.0,    # Lysine: fully protonated (pKa ~10.8)
    "ASP": -1.0,    # Aspartate: fully deprotonated (pKa ~3.9)
    "GLU": -1.0,    # Glutamate: fully deprotonated (pKa ~4.2)
    "HIS": +0.1     # Histidine: partially protonated (pKa ~6.0)
}

# Legacy sets for backward compatibility
POSITIVE = {"ARG", "LYS"}
NEGATIVE = {"ASP", "GLU"}
HIS_SET = {"HIS", "HSD", "HSE", "HSP"}

# RCSB PDB download template
PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{}.pdb"

# Default parameters
DEFAULT_CUTOFF = 5.0
NEIGHBOR_SURFACE_CUTOFF = 8.0
NEIGHBOR_BURIED_THRESHOLD = 25
DEFAULT_BATCH_SIZE = 10

print("🧬 Enhanced PDB Interface Charge Analysis")
print("✅ All modules imported successfully")

# Core utility functions
def download_pdb(pdb_id, outdir):
    """Download a PDB file from RCSB and cache locally."""
    pdb_id = pdb_id.strip().lower()
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{pdb_id}.pdb")
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return fn
    url = PDB_URL_TEMPLATE.format(pdb_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(fn, "wb") as fh:
        fh.write(r.content)
    return fn

def parse_structure(pdb_file, pdb_id):
    """Parse a PDB file into a Bio.PDB Structure object."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    return structure

def is_aa_residue(res):
    """Check if residue is a standard amino acid."""
    return PDB.is_aa(res, standard=True)

def get_surface_residues_by_neighbors(structure, model_index=0, neighbor_cutoff=NEIGHBOR_SURFACE_CUTOFF, buried_threshold=NEIGHBOR_BURIED_THRESHOLD):
    """Heuristic fallback for surface detection."""
    model = structure[model_index]
    atoms = Selection.unfold_entities(model, "A")
    ns = NeighborSearch(list(atoms))
    res_counts = {}
    
    for chain in model:
        for res in chain:
            if not is_aa_residue(res):
                continue
            if "CA" in res:
                coord = res["CA"].get_coord()
            else:
                coords = [a.get_coord() for a in res.get_unpacked_list()]
                coord = sum(coords) / len(coords)
            nearby_atoms = ns.search(coord, neighbor_cutoff)
            heavy_count = sum(1 for a in nearby_atoms if getattr(a, "element", "").upper() != "H")
            res_counts[(chain.id, res.id[1])] = heavy_count
    
    surface = set(k for k, cnt in res_counts.items() if cnt < buried_threshold)
    return surface

def get_interface_residue_sets(structure, pdb_file=None, cutoff=DEFAULT_CUTOFF,
                               require_surface=True, use_dssp=False):
    """Detect chain-chain interfaces in a structure."""
    model = structure[0]
    atoms = Selection.unfold_entities(model, "A")
    ns = NeighborSearch(list(atoms))
    chains = [c for c in model if any(is_aa_residue(r) for r in c)]
    
    surface_set = None
    if require_surface:
        surface_set = get_surface_residues_by_neighbors(structure)

    interfaces = []
    for i in range(len(chains)):
        ca = chains[i]
        for j in range(i + 1, len(chains)):
            cb = chains[j]
            atoms_a = Selection.unfold_entities(ca, "A")
            
            contact_pairs = []
            for atom in atoms_a:
                close_atoms = ns.search(atom.get_coord(), cutoff)
                for close_atom in close_atoms:
                    if close_atom.get_parent().get_parent().id != cb.id:
                        continue
                    contact_pairs.append((atom.get_parent(), close_atom.get_parent()))
            
            if not contact_pairs:
                continue
                
            residues_a = set()
            residues_b = set()
            for ra, rb in contact_pairs:
                if not is_aa_residue(ra) or not is_aa_residue(rb):
                    continue
                key_a = (ra.get_parent().id, ra.id[1])
                key_b = (rb.get_parent().id, rb.id[1])
                
                if surface_set is not None:
                    if (key_a not in surface_set) and (key_b not in surface_set):
                        continue
                        
                residues_a.add((ra.get_parent().id, ra.id[1], ra.get_resname()))
                residues_b.add((rb.get_parent().id, rb.id[1], rb.get_resname()))
            
            if residues_a and residues_b:
                interfaces.append({
                    "chain_a": ca.id,
                    "chain_b": cb.id,
                    "residues_a": residues_a,
                    "residues_b": residues_b
                })
    return interfaces

def extract_all_residues_from_structure(structure):
    """Extract all amino acid residues from the entire protein structure."""
    all_residues = {}
    residue_details = []
    
    model = structure[0]
    for chain in model:
        chain_residues = []
        for residue in chain:
            if is_aa_residue(residue):
                res_info = {
                    'chain': chain.id,
                    'residue_number': residue.id[1],
                    'residue_name': residue.get_resname(),
                    'insertion_code': residue.id[2] if residue.id[2] != ' ' else None
                }
                chain_residues.append(res_info)
                residue_details.append(res_info)
        
        if chain_residues:
            all_residues[chain.id] = chain_residues
    
    return all_residues, residue_details

def analyze_protein_composition(residue_details):
    """Analyze the complete amino acid composition of the protein."""
    composition = Counter()
    charge_analysis = {
        'total_positive': 0,
        'total_negative': 0,
        'total_neutral': 0,
        'total_charge': 0.0,
        'charged_residues': [],
        'chain_charges': {}
    }
    
    for residue in residue_details:
        res_name = residue['residue_name'].upper()
        chain = residue['chain']
        
        composition[res_name] += 1
        
        if res_name in RESIDUE_CHARGES:
            charge = RESIDUE_CHARGES[res_name]
            charge_analysis['total_charge'] += charge
            
            if charge > 0:
                charge_analysis['total_positive'] += charge
            elif charge < 0:
                charge_analysis['total_negative'] += abs(charge)
            
            charge_analysis['charged_residues'].append({
                'chain': chain,
                'position': residue['residue_number'], 
                'residue': res_name,
                'charge': charge
            })
        else:
            charge_analysis['total_neutral'] += 1
    
    # Calculate per-chain charges
    for residue in residue_details:
        chain = residue['chain']
        res_name = residue['residue_name'].upper()
        
        if chain not in charge_analysis['chain_charges']:
            charge_analysis['chain_charges'][chain] = {
                'total_charge': 0.0,
                'positive_charge': 0.0,
                'negative_charge': 0.0,
                'residue_count': 0,
                'charged_residues': []
            }
        
        charge_analysis['chain_charges'][chain]['residue_count'] += 1
        
        if res_name in RESIDUE_CHARGES:
            charge = RESIDUE_CHARGES[res_name]
            charge_analysis['chain_charges'][chain]['total_charge'] += charge
            
            if charge > 0:
                charge_analysis['chain_charges'][chain]['positive_charge'] += charge
            elif charge < 0:
                charge_analysis['chain_charges'][chain]['negative_charge'] += abs(charge)
                
            charge_analysis['chain_charges'][chain]['charged_residues'].append({
                'position': residue['residue_number'],
                'residue': res_name,
                'charge': charge
            })
    
    return composition, charge_analysis, {}

def count_charges_in_res_list(residues, his_as_positive=False):
    """Count actual charges (not just residue counts) in a residue set."""
    total_charge = 0.0
    positive_charge = 0.0
    negative_charge = 0.0
    counts = Counter()
    
    for (chain, resseq, resname) in residues:
        rn = resname.strip().upper()
        counts[rn] += 1
        
        if rn in RESIDUE_CHARGES:
            charge = RESIDUE_CHARGES[rn]
            if rn == "HIS" and not his_as_positive:
                charge = 0.0
            
            total_charge += charge
            if charge > 0:
                positive_charge += charge
            elif charge < 0:
                negative_charge += abs(charge)
    
    return {
        "pos_charge": positive_charge,
        "neg_charge": negative_charge, 
        "net_charge": total_charge,
        "pos_count": len([r for r in residues if r[2].upper() in POSITIVE or (r[2].upper() in HIS_SET and his_as_positive)]),
        "neg_count": len([r for r in residues if r[2].upper() in NEGATIVE]),
        "counts": dict(counts)
    }

def analyze_pdb(pdb_file, pdb_id, cutoff=DEFAULT_CUTOFF,
                require_surface=True, use_dssp=False, his_as_positive=False):
    """Analyze interfaces in a PDB and compute per-interface charge stats + full protein composition."""
    structure = parse_structure(pdb_file, pdb_id)
    
    # Extract ALL residues from the protein
    all_residues, residue_details = extract_all_residues_from_structure(structure)
    composition, charge_analysis, chain_counts = analyze_protein_composition(residue_details)
    
    # Get interface-specific data
    interfaces = get_interface_residue_sets(structure, pdb_file=pdb_file, cutoff=cutoff,
                                            require_surface=require_surface, use_dssp=use_dssp)
    interface_stats = []
    
    for iface in interfaces:
        a_stats = count_charges_in_res_list(iface["residues_a"], his_as_positive=his_as_positive)
        b_stats = count_charges_in_res_list(iface["residues_b"], his_as_positive=his_as_positive)
        
        charge_imbalance = abs(a_stats["net_charge"] - b_stats["net_charge"])
        signed_charge_imbalance = a_stats["net_charge"] - b_stats["net_charge"]
        total_residues = len(iface["residues_a"]) + len(iface["residues_b"])
        total_charge = a_stats["net_charge"] + b_stats["net_charge"]
        
        interface_stats.append({
            "chain_a": iface["chain_a"],
            "chain_b": iface["chain_b"],
            "residues_a_count": len(iface["residues_a"]),
            "residues_b_count": len(iface["residues_b"]),
            "charge_a": a_stats["net_charge"],
            "pos_charge_a": a_stats["pos_charge"],
            "neg_charge_a": a_stats["neg_charge"],
            "charge_b": b_stats["net_charge"],
            "pos_charge_b": b_stats["pos_charge"], 
            "neg_charge_b": b_stats["neg_charge"],
            "pos_count_a": a_stats["pos_count"],
            "neg_count_a": a_stats["neg_count"],
            "pos_count_b": b_stats["pos_count"],
            "neg_count_b": b_stats["neg_count"],
            "charge_imbalance": charge_imbalance,
            "signed_charge_imbalance": signed_charge_imbalance,
            "total_interface_charge": total_charge,
            "total_interface_residues": total_residues
        })
    
    return interface_stats, composition, charge_analysis, all_residues

def aggregate_interface_stats(interface_stats):
    """Aggregate interface stats into per-PDB summary using actual charges."""
    if not interface_stats:
        return {
            "interface_count": 0,
            "max_charge_imbalance": 0.0,
            "total_protein_charge": 0.0,
            "avg_interface_charge": 0.0,
            "avg_charge_imbalance": 0.0,
            "avg_total_interface_residues": 0.0
        }
    
    interface_count = len(interface_stats)
    max_charge_imbalance = max(i["charge_imbalance"] for i in interface_stats)
    total_protein_charge = sum(i["total_interface_charge"] for i in interface_stats)
    avg_interface_charge = total_protein_charge / interface_count if interface_count > 0 else 0
    avg_charge_imbalance = sum(i["charge_imbalance"] for i in interface_stats) / interface_count
    avg_total_interface_residues = sum(i["total_interface_residues"] for i in interface_stats) / interface_count
    
    best_iface = max(interface_stats, key=lambda x: abs(x["signed_charge_imbalance"]))
    
    return {
        "interface_count": interface_count,
        "max_charge_imbalance": max_charge_imbalance,
        "total_protein_charge": total_protein_charge,
        "avg_interface_charge": avg_interface_charge,
        "avg_charge_imbalance": avg_charge_imbalance,
        "avg_total_interface_residues": avg_total_interface_residues,
        "best_signed_charge_imbalance": best_iface["signed_charge_imbalance"],
        "best_chain_pair": f"{best_iface['chain_a']}-{best_iface['chain_b']}",
        "best_charge_a": best_iface["charge_a"],
        "best_charge_b": best_iface["charge_b"]
    }

def filter_by_charge_distribution(df, pdb_column):
    """Filter dataframe to only include proteins with red or orange charge distribution."""
    if 'charge_distribution' not in df.columns:
        print("⚠️ No 'charge_distribution' column found in CSV")
        print("Available columns:", list(df.columns))
        use_all = input("Process all proteins anyway? (Y/n): ").strip().lower()
        if use_all == 'n':
            return []
        else:
            return df[pdb_column].dropna().astype(str).tolist()
    
    filtered_df = df[df['charge_distribution'].isin(['red', 'orange', 'Red', 'Orange', 'RED', 'ORANGE','Purple','purple'])]
    
    if len(filtered_df) == 0:
        print("❌ No proteins found with red or orange charge distribution")
        print("Available charge distributions:", df['charge_distribution'].value_counts().to_dict())
        return []
    
    pdb_ids = filtered_df[pdb_column].dropna().astype(str).tolist()
    print(f"🔍 Filtered to {len(pdb_ids)} proteins with red/orange/purple charge distribution")
    print(f"   Red: {len(filtered_df[filtered_df['charge_distribution'].str.lower() == 'red'])}")
    print(f"   Orange: {len(filtered_df[filtered_df['charge_distribution'].str.lower() == 'orange'])}")
    print(f"   Purple: {len(filtered_df[filtered_df['charge_distribution'].str.lower() == 'purple'])}")
    
    return pdb_ids

# Enhanced batch processing class
class EnhancedPDBAnalyzer:
    def __init__(self, workdir="./pdb_cache"):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.results = []
        self.detailed_results = []
        
    def analyze_single_pdb(self, pdb_id, **kwargs):
        """Analyze a single PDB with detailed output including full protein composition."""
        print(f"\n🔍 Analyzing single PDB: {pdb_id.upper()}")
        print("=" * 50)
        
        try:
            start_time = time.time()
            pdb_file = download_pdb(pdb_id, self.workdir)
            print(f"✅ Downloaded: {pdb_file}")
            
            interface_stats, composition, charge_analysis, all_residues = analyze_pdb(pdb_file, pdb_id, **kwargs)
            summary = aggregate_interface_stats(interface_stats)
            
            result = {
                "pdb_id": pdb_id.upper(),
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - start_time,
                **summary,
                "total_residues": sum(composition.values()),
                "total_protein_charge": charge_analysis['total_charge'],
                "total_positive_charge": charge_analysis['total_positive'],
                "total_negative_charge": charge_analysis['total_negative'],
                "charged_residue_count": len(charge_analysis['charged_residues']),
                "amino_acid_composition": dict(composition)
            }
            
            # Display results with full protein information
            print(f"\n🧬 Full Protein Composition for {pdb_id.upper()}:")
            print("=" * 50)
            
            print(f"📊 Amino Acid Composition ({result['total_residues']} total residues):")
            sorted_composition = sorted(composition.items(), key=lambda x: x[1], reverse=True)
            for aa, count in sorted_composition:
                percentage = (count / result['total_residues']) * 100
                charge_indicator = ""
                if aa in RESIDUE_CHARGES:
                    charge_indicator = f" ({RESIDUE_CHARGES[aa]:+.1f})"
                print(f"  {aa}: {count:3d} ({percentage:4.1f}%){charge_indicator}")
            
            print(f"\n⚡ Charge Analysis:")
            print(f"  Total protein charge: {charge_analysis['total_charge']:+.2f}")
            print(f"  Positive charge: +{charge_analysis['total_positive']:.2f}")
            print(f"  Negative charge: -{charge_analysis['total_negative']:.2f}")
            print(f"  Charged residues: {len(charge_analysis['charged_residues'])}/{result['total_residues']} ({len(charge_analysis['charged_residues'])/result['total_residues']*100:.1f}%)")
            
            return result, interface_stats, composition, charge_analysis
            
        except Exception as e:
            error_result = {
                "pdb_id": pdb_id.upper(),
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
            print(f"❌ Error analyzing {pdb_id.upper()}: {e}")
            return error_result, [], {}, {}
    
    def save_incremental_results(self, output_file, summary_results, detailed_results=None):
        """Save results incrementally to avoid data loss."""
        try:
            df_summary = pd.DataFrame(summary_results)
            df_summary.to_csv(output_file, index=False)
            print(f"💾 Saved {len(summary_results)} results to {output_file}")
            
            if detailed_results:
                detailed_file = output_file.replace('.csv', '_detailed.csv')
                df_detailed = pd.DataFrame(detailed_results)
                df_detailed.to_csv(detailed_file, index=False)
                print(f"💾 Saved {len(detailed_results)} detailed results to {detailed_file}")
            
            return True
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def load_existing_results(self, output_file):
        """Load existing results to enable resume functionality."""
        completed_ids = set()
        existing_results = []
        
        if os.path.exists(output_file):
            try:
                df = pd.read_csv(output_file)
                existing_results = df.to_dict('records')
                completed_ids = {str(row['pdb_id']).lower() for row in existing_results if 'error' not in row}
                print(f"📂 Found {len(existing_results)} existing results ({len(completed_ids)} successful)")
            except Exception as e:
                print(f"⚠️ Error loading existing results: {e}")
        
        return existing_results, completed_ids

# User interaction functions
def get_user_parameters():
    """Interactive parameter configuration."""
    print("\n⚙️ Analysis Parameters Configuration")
    print("=" * 40)
    
    cutoff_input = input(f"Interface cutoff distance (Å) [default: {DEFAULT_CUTOFF}]: ").strip()
    cutoff = float(cutoff_input) if cutoff_input else DEFAULT_CUTOFF
    
    surface_input = input("Require surface residues? (y/N): ").strip().lower()
    require_surface = surface_input == 'y'
    
    dssp_input = input("Use DSSP for surface detection? (y/N): ").strip().lower()
    use_dssp = dssp_input == 'y' and has_dssp
    if dssp_input == 'y' and not has_dssp:
        print("⚠️ DSSP not available, will use heuristic method")
    
    his_input = input("Count histidine as positive? (y/N): ").strip().lower()
    his_as_positive = his_input == 'y'
    
    return {
        'cutoff': cutoff,
        'require_surface': require_surface,
        'use_dssp': use_dssp,
        'his_as_positive': his_as_positive
    }

def select_run_mode():
    """Interactive run mode selection."""
    print("\n🚀 Select Analysis Mode")
    print("=" * 30)
    print("1. Single PDB Analysis")
    print("2. Batch Processing (N proteins)")
    print("3. Full CSV Processing (with resume)")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect mode (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def run_single_mode(analyzer, params):
    """Handle single PDB analysis mode."""
    pdb_id = input("Enter PDB ID: ").strip()
    if not pdb_id:
        print("❌ No PDB ID provided")
        return
    
    result, detailed, composition, charge_analysis = analyzer.analyze_single_pdb(pdb_id, **params)
    
    save_choice = input("\nSave results to file? (Y/n): ").strip().lower()
    if save_choice != 'n':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"single_analysis_{pdb_id}_{timestamp}.csv"
        
        detailed_rows = []
        for i, iface in enumerate(detailed):
            row = {"pdb_id": pdb_id.upper(), "interface_index": i}
            row.update(iface)
            detailed_rows.append(row)
        
        analyzer.save_incremental_results(output_file, [result], detailed_rows)

def run_batch_mode(analyzer, params):
    """Handle batch processing mode with charge distribution filtering."""
    batch_input = input(f"Batch size [default: {DEFAULT_BATCH_SIZE}]: ").strip()
    batch_size = int(batch_input) if batch_input.isdigit() else DEFAULT_BATCH_SIZE
    
    print("\nEnter PDB IDs (multiple options):")
    print("1. Enter comma-separated list")
    print("2. Load from CSV file (filter by charge_distribution)")
    
    input_choice = input("Select input method (1/2): ").strip()
    
    pdb_ids = []
    if input_choice == '1':
        pdb_input = input("Enter PDB IDs (comma-separated): ").strip()
        pdb_ids = [pid.strip() for pid in pdb_input.split(',') if pid.strip()]
    elif input_choice == '2':
        csv_file = input("Enter CSV file path: ").strip()
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                pdb_col = None
                for col in df.columns:
                    if col.lower() in ['pdb_id', 'pdbid', 'pdb', 'pdb_code_key']:
                        pdb_col = col
                        break
                
                if pdb_col:
                    pdb_ids = filter_by_charge_distribution(df, pdb_col)
                    if pdb_ids:
                        print(f"✅ Loaded {len(pdb_ids)} PDB IDs from {csv_file}")
                    else:
                        return
                else:
                    print("❌ No PDB ID column found in CSV")
                    return
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
                return
        else:
            print("❌ CSV file not found")
            return
    else:
        # Handle case where user enters file path directly instead of choosing option
        if input_choice.endswith('.csv') and os.path.exists(input_choice):
            print(f"📁 Detected CSV file path: {input_choice}")
            csv_file = input_choice
            try:
                df = pd.read_csv(csv_file)
                pdb_col = None
                for col in df.columns:
                    if col.lower() in ['pdb_id', 'pdbid', 'pdb', 'pdb_code_key']:
                        pdb_col = col
                        break
                
                if pdb_col:
                    pdb_ids = filter_by_charge_distribution(df, pdb_col)
                    if pdb_ids:
                        print(f"✅ Loaded {len(pdb_ids)} PDB IDs from {csv_file}")
                    else:
                        return
                else:
                    print("❌ No PDB ID column found in CSV")
                    return
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
                return
        else:
            print("❌ Invalid input. Please enter 1 or 2, or provide a valid CSV file path.")
            return
    
    if not pdb_ids:
        print("❌ No PDB IDs provided")
        return
    
    if len(pdb_ids) > batch_size:
        print(f"📊 Found {len(pdb_ids)} filtered PDB IDs. Processing first {batch_size} in this batch.")
        pdb_ids = pdb_ids[:batch_size]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"batch_analysis_filtered_{timestamp}.csv"
    
    results = []
    detailed_results = []
    
    print(f"\n🚀 Starting batch processing of {len(pdb_ids)} proteins (red/orange charge distribution)...")
    
    for i, pdb_id in enumerate(pdb_ids, 1):
        print(f"\n[{i}/{len(pdb_ids)}] Processing {pdb_id.upper()}")
        
        try:
            start_time = time.time()
            pdb_file = download_pdb(pdb_id, analyzer.workdir)
            interface_stats, composition, charge_analysis, all_residues = analyze_pdb(pdb_file, pdb_id, **params)
            summary = aggregate_interface_stats(interface_stats)
            
            result = {
                "pdb_id": pdb_id.upper(),
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - start_time,
                **summary,
                "total_residues": sum(composition.values()),
                "total_protein_charge": charge_analysis['total_charge'],
                "amino_acid_composition": dict(composition)
            }
            results.append(result)
            
            # Display detailed chain information for each protein
            print(f"\n  Chain Details for {pdb_id.upper()}:")
            print(f"    Total residues: {sum(composition.values())}")
            print(f"    Total protein charge: {charge_analysis['total_charge']:+.2f}")
            
            # Show per-chain breakdown
            for chain_id, chain_data in charge_analysis['chain_charges'].items():
                print(f"    Chain {chain_id}:")
                print(f"      Residues: {chain_data['residue_count']}")
                print(f"      Net charge: {chain_data['total_charge']:+.2f}")
                print(f"      Positive: +{chain_data['positive_charge']:.2f}")
                print(f"      Negative: -{chain_data['negative_charge']:.2f}")
                print(f"      Charged residues: {len(chain_data['charged_residues'])}")
                
                # Show charged residue types and positions for each chain
                if chain_data['charged_residues']:
                    by_type = {}
                    for res in chain_data['charged_residues']:
                        if res['residue'] not in by_type:
                            by_type[res['residue']] = []
                        by_type[res['residue']].append(res['position'])
                    
                    print(f"      Charged residue positions:")
                    for res_type in sorted(by_type.keys()):
                        positions = sorted(by_type[res_type])
                        charge = RESIDUE_CHARGES[res_type]
                        # Show first 10 positions to avoid too much output
                        if len(positions) > 10:
                            pos_str = f"{positions[:10]}... (+{len(positions)-10} more)"
                        else:
                            pos_str = str(positions)
                        print(f"        {res_type} ({charge:+.1f}): {pos_str}")
            
            # Add chain composition to result for CSV output
            chain_details = {}
            for chain_id, chain_data in charge_analysis['chain_charges'].items():
                chain_details[f"chain_{chain_id}_residues"] = chain_data['residue_count']
                chain_details[f"chain_{chain_id}_charge"] = chain_data['total_charge']
                chain_details[f"chain_{chain_id}_positive"] = chain_data['positive_charge']
                chain_details[f"chain_{chain_id}_negative"] = chain_data['negative_charge']
            
            result.update(chain_details)
            
            for j, iface in enumerate(interface_stats):
                row = {"pdb_id": pdb_id.upper(), "interface_index": j}
                row.update(iface)
                detailed_results.append(row)
            
            print(f"✅ {pdb_id.upper()}: {summary['interface_count']} interfaces, total protein charge: {charge_analysis['total_charge']:+.2f}")
            print(f"    Composition summary: {len([aa for aa, count in composition.most_common(5)])} most common AAs: {', '.join([f'{aa}({count})' for aa, count in composition.most_common(5)])}")
            
            if i % 5 == 0:
                analyzer.save_incremental_results(output_file, results, detailed_results)
                
        except Exception as e:
            error_result = {
                "pdb_id": pdb_id.upper(),
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
            results.append(error_result)
            print(f"❌ {pdb_id.upper()}: {e}")
    
    analyzer.save_incremental_results(output_file, results, detailed_results)
    
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n🎉 Batch Analysis Complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {output_file}")

def run_full_mode(analyzer, params):
    """Handle full CSV processing with resume capability and charge distribution filtering."""
    csv_file = input("Enter CSV file path: ").strip()
    if not os.path.exists(csv_file):
        print("❌ CSV file not found")
        return
    
    # Load CSV and detect PDB column
    try:
        df = pd.read_csv(csv_file)
        pdb_col = None
        for col in df.columns:
            if col.lower() in ['pdb_id', 'pdbid', 'pdb', 'pdb_code_key']:
                pdb_col = col
                break
        
        if not pdb_col:
            print("❌ No PDB ID column found in CSV")
            return
        
        # Filter by charge distribution
        all_pdb_ids = filter_by_charge_distribution(df, pdb_col)
        if not all_pdb_ids:
            return
            
        print(f"✅ Found {len(all_pdb_ids)} PDB IDs with red/orange/purple charge distribution")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Set up output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"full_analysis_filtered_{timestamp}.csv"
    
    # Check for resume capability
    resume_choice = input("Resume from previous run if available? (Y/n): ").strip().lower()
    existing_results, completed_ids = [], set()
    
    if resume_choice != 'n':
        existing_results, completed_ids = analyzer.load_existing_results(output_file)
    
    # Filter out completed IDs
    remaining_ids = [pid for pid in all_pdb_ids if pid.lower() not in completed_ids]
    
    if not remaining_ids:
        print("All filtered PDB IDs have already been processed!")
        return
    
    print(f"Processing {len(remaining_ids)} remaining PDB IDs with red/orange/purple charge distribution...")
    
    # Get batch size for incremental saves
    batch_input = input(f"Save frequency (every N proteins) [default: 20]: ").strip()
    save_frequency = int(batch_input) if batch_input.isdigit() else 20
    
    # Process all remaining
    results = existing_results.copy()
    detailed_results = []
    
    print(f"\nStarting full processing of filtered proteins...")
    
    for i, pdb_id in enumerate(remaining_ids, 1):
        print(f"\n[{i}/{len(remaining_ids)}] Processing {pdb_id.upper()}")
        
        try:
            start_time = time.time()
            pdb_file = download_pdb(pdb_id, analyzer.workdir)
            interface_stats, composition, charge_analysis, all_residues = analyze_pdb(pdb_file, pdb_id, **params)
            summary = aggregate_interface_stats(interface_stats)
            
            result = {
                "pdb_id": pdb_id.upper(),
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - start_time,
                **summary,
                "total_residues": sum(composition.values()),
                "total_protein_charge": charge_analysis['total_charge'],
                "amino_acid_composition": dict(composition)
            }
            
            # Add chain details to result
            chain_details = {}
            for chain_id, chain_data in charge_analysis['chain_charges'].items():
                chain_details[f"chain_{chain_id}_residues"] = chain_data['residue_count']
                chain_details[f"chain_{chain_id}_charge"] = chain_data['total_charge']
                chain_details[f"chain_{chain_id}_positive"] = chain_data['positive_charge']
                chain_details[f"chain_{chain_id}_negative"] = chain_data['negative_charge']
            
            result.update(chain_details)
            results.append(result)
            
            # Add detailed results
            for j, iface in enumerate(interface_stats):
                row = {"pdb_id": pdb_id.upper(), "interface_index": j}
                row.update(iface)
                detailed_results.append(row)
            
            print(f"✅ {pdb_id.upper()}: {summary['interface_count']} interfaces, charge: {charge_analysis['total_charge']:+.2f}")
            
            # Save incrementally
            if i % save_frequency == 0:
                analyzer.save_incremental_results(output_file, results, detailed_results)
                print(f"💾 Incremental save completed ({i}/{len(remaining_ids)})")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Process interrupted by user!")
            analyzer.save_incremental_results(output_file, results, detailed_results)
            return
            
        except Exception as e:
            error_result = {
                "pdb_id": pdb_id.upper(),
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
            results.append(error_result)
            print(f"❌ {pdb_id.upper()}: {e}")
    
    # Final save
    analyzer.save_incremental_results(output_file, results, detailed_results)
    
    # Final summary
    total_processed = len([r for r in results if r not in existing_results])
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n🎉 Full Analysis Complete!")
    print(f"📊 Total filtered proteins: {len(all_pdb_ids)}")
    print(f"🆕 Newly processed: {total_processed}")
    print(f"✅ Total successful: {successful}")
    print(f"❌ Total failed: {failed}")
    print(f"📁 Results saved to: {output_file}")
    
    if successful > 0:
        success_results = [r for r in results if 'error' not in r and 'total_protein_charge' in r]
        if success_results:
            avg_charge = sum(r['total_protein_charge'] for r in success_results) / len(success_results)
            max_charge = max(r['total_protein_charge'] for r in success_results)
            min_charge = min(r['total_protein_charge'] for r in success_results)
            
            print(f"\n📊 Charge Statistics (Red/Orange proteins):")
            print(f"   Average protein charge: {avg_charge:+.2f}")
            print(f"   Maximum protein charge: {max_charge:+.2f}")
            print(f"   Minimum protein charge: {min_charge:+.2f}")

def main():
    """Main interactive application."""
    print("🧬 Enhanced PDB Interface Charge Analysis")
    print("=" * 50)
    print("Features:")
    print("✅ Single PDB analysis with detailed output")
    print("✅ Batch processing with red/orange filtering")
    print("✅ Full CSV processing with resume capability")
    print("✅ Complete amino acid composition analysis")
    print("✅ Actual charge calculations (not just counts)")
    
    workdir_input = input(f"\nWork directory [default: ./pdb_cache]: ").strip()
    workdir = workdir_input if workdir_input else "./pdb_cache"
    
    analyzer = EnhancedPDBAnalyzer(workdir)
    
    while True:
        try:
            mode = select_run_mode()
            
            if mode == 4:
                print("👋 Thanks for using Enhanced PDB Analysis!")
                break
            
            params = get_user_parameters()
            
            print(f"\n📋 Analysis Configuration:")
            print(f"  🎯 Interface cutoff: {params['cutoff']} Å")
            print(f"  🌊 Require surface: {params['require_surface']}")
            print(f"  🔬 Use DSSP: {params['use_dssp']}")
            print(f"  ⚗️  HIS as positive: {params['his_as_positive']}")
            
            if mode == 1:
                run_single_mode(analyzer, params)
            elif mode == 2:
                run_batch_mode(analyzer, params)
            elif mode == 3:
                run_full_mode(analyzer, params)
            
            continue_choice = input("\nRun another analysis? (Y/n): ").strip().lower()
            if continue_choice == 'n':
                print("👋 Thanks for using Enhanced PDB Analysis!")
                break
                
        except KeyboardInterrupt:
            print("\n🛑 Process interrupted by user!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue_choice = input("Continue with analysis? (Y/n): ").strip().lower()
            if continue_choice == 'n':
                break

if __name__ == "__main__":
    main()