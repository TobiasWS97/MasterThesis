import os

  
def write_mdp(temp):
           
           with open("nvt.mdp","w") as writing:
            writing.write("""
integrator = md
nsteps     = 15000000
dt         = 0.001

nstxout       = 200
nstenergy     = 200

continuation         = yes
constraint_algorithm = lincs
constraints          = h-bonds
lincs_iter           = 1
lincs_order          = 4

cutoff-scheme = Verlet
ns_type       = grid
nstlist       = 10
rcoulomb      = 0.9
rvdw          = 0.9
DispCorr      = EnerPres

coulombtype    = PME
pme_order      = 4
fourierspacing = 0.12

tcoupl         = v-rescale
tc-grps        = System
tau_t          = 0.1 
ref_t          = {a}

pcoupl         = no

pbc           = xyz

""".format(a=temp)) # {a} = temperature from csv file

pwd=os.getcwd()
with open('LHC_cat_1000.csv', 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        
        temp = fields[0] # temperatures
        conc = fields[1] # concentrations
        if conc == '1.0':
            os.chdir(pwd+"/100")
        elif conc == '2.0':
            os.chdir(pwd+"/75")
        elif conc == '3.0':
            os.chdir(pwd+"/66")
        
        write_mdp(temp)
        name = temp.replace('.','_')+"_"+str(int(float(conc)))

        os.system("gmx grompp -f nvt.mdp -c box.gro -p box.top -o "+name+".tpr -maxwarn 2")
        os.system("gmx mdrun -v -deffnm "+name+"")
        os.system("echo 0 | gmx traj -f "+name+".trr -s "+name+".tpr -nojump -oxt "+name+".xtc")
        os.system("rm *.cpt *.edr *.mdp *.trr *.log")
        os.chdir(pwd)





