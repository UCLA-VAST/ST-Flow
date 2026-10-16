#!/usr/bin/env python3
import subprocess
import os
import sys
from rich.console import Console
from rich.table import Table

console = Console()

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def main():
    # Node info
    # %n name, %t state, %m memory, %c cpus, %O load, %C CPUS(A/I/O/T), %e free_mem
    out = run('sinfo -o "%n|%t|%m|%c|%O|%C|%e" -h')
    
    if not out:
        console.print("[red]Error: Could not retrieve node information via sinfo.[/red]")
        return

    nodes = []
    for line in out.strip().split('\n'):
        if not line: continue
        parts = line.split('|')
        if len(parts) < 7: continue
        nodes.append({
            'name': parts[0],
            'state': parts[1],
            'mem': parts[2],
            'cpus': parts[3],
            'load': parts[4],
            'cpu_alloc': parts[5],
            'free_mem': parts[6]
        })

    # User jobs info
    jobs_out = run('squeue -u $USER -o "%i|%j|%T|%N" -h')
    node_jobs = {}
    for line in jobs_out.strip().split('\n'):
        if not line: continue
        parts = line.split('|')
        if len(parts) < 4: continue
        jid, name, status, node_list = parts
        
        # Handle cases where node_list might be multiple nodes or empty
        if not node_list or node_list == "(null)": continue
        
        # Slurm might return c0[1-2] format, but for simple per-node check we just split if comma
        for node in node_list.split(','):
            if node not in node_jobs: node_jobs[node] = []
            node_jobs[node].append(f"{jid}({name})")

    table = Table(title="Slurm Cluster Health Monitor")
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("State", style="magenta")
    table.add_column("Load", justify="right")
    table.add_column("CPUs (Alloc/Total)", justify="right")
    table.add_column("Mem (Free/Total)", justify="right")
    table.add_column("My Active Jobs", style="yellow")

    for n in sorted(nodes, key=lambda x: x['name']):
        my_jobs = ", ".join(node_jobs.get(n['name'], ["-"]))
        
        # Color coding for state
        state = n['state'].lower()
        state_display = state
        if 'idle' in state: 
            state_display = f"[green]{state}[/green]"
        elif 'down' in state or 'drain' in state or 'maint' in state: 
            state_display = f"[bold white on red]{state}[/bold white on red]"
        elif 'mix' in state: 
            state_display = f"[yellow]{state}[/yellow]"
        elif 'alloc' in state:
            state_display = f"[blue]{state}[/blue]"
        
        # Color coding for load
        load_val = n['load']
        load_str = load_val
        try:
            if load_val != "N/A":
                load = float(load_val)
                total_cpus = int(n['cpus'])
                if load > total_cpus:
                    load_str = f"[bold red]{load_val}[/bold red]"
                elif load > total_cpus * 0.8:
                    load_str = f"[yellow]{load_val}[/yellow]"
        except ValueError:
            pass

        # CPUs A/T
        try:
            alloc_cpus = n['cpu_alloc'].split('/')[0]
            cpu_str = f"{alloc_cpus}/{n['cpus']}"
        except:
            cpu_str = n['cpu_alloc']
        
        # Mem F/T (MB -> GB)
        try:
            free = int(n['free_mem']) // 1024
            total = int(n['mem']) // 1024
            mem_str = f"{free}G/{total}G"
            # Highlight if free memory is very low (less than 5% or 4GB)
            if free < 4:
                mem_str = f"[bold red]{mem_str}[/bold red]"
        except:
            mem_str = f"{n['free_mem']}/{n['mem']}"

        table.add_row(
            n['name'],
            state_display,
            load_str,
            cpu_str,
            mem_str,
            my_jobs
        )

    console.print(table)
    
    # Check for unresponsive nodes (running jobs but 0 or near-0 load)
    console.print("\n[bold]Potential Unresponsive Nodes (Jobs Running but Low Load):[/bold]")
    warnings = 0
    for n in nodes:
        if n['name'] in node_jobs:
            try:
                load = float(n['load'])
                if load < 0.1: # Load is effectively zero
                    console.print(f"  [red]Warning: {n['name']} has jobs assigned but Load is {n['load']}. Might be hanging.[/red]")
                    warnings += 1
            except:
                pass
    if warnings == 0:
        console.print("  [green]None detected based on current load heuristics.[/green]")

if __name__ == "__main__":
    main()
