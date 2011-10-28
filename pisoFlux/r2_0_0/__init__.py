#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV, Andrey SIMURZIN
##


#---------------------------------------------------------------------------
from Foam import ref, man


#---------------------------------------------------------------------------
def _createFields( runTime, mesh ):
    
    ref.ext_Info() << "Reading field p\n" << ref.nl
    p = man.volScalarField( man.IOobject( ref.word( "p" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )
    
    ref.ext_Info() << "Reading field U\n" << ref.nl
    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )
    
    phi = man.createPhi( runTime, mesh, U )
    
    pRefCell = 0
    pRefValue = 0.0
    
    pRefCell, pRefValue = ref.setRefCell( p, mesh.solutionDict().subDict( ref.word( "PISO" ) ), pRefCell, pRefValue )
    
    laminarTransport = man.singlePhaseTransportModel( U, phi )
    
    turbulence = man.incompressible.turbulenceModel.New( U, phi, laminarTransport )

    return p, U, phi, turbulence, pRefCell, pRefValue, laminarTransport


#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMesh( runTime )

    p, U, phi, turbulence, pRefCell, pRefValue, laminarTransport = _createFields( runTime, mesh )
    
    cumulativeContErr = ref.initContinuityErrs()

    ref.ext_Info() << "\nStarting time loop\n" <<ref.nl
    
    while runTime.loop() :
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl

        piso, nCorr, nNonOrthCorr, momentumPredictor, transonic, nOuterCorr = ref.readPISOControls( mesh )

        CoNum, meanCoNum = ref.CourantNo( mesh, phi, runTime )

        # Pressure-velocity PISO corrector

        # Momentum predictor

        # The initial C++ expression does not work properly, because of
        #  1. turbulence.divDevRhoReff( U ) - changes values for the U boundaries
        #  2. the order of expression arguments computation differs with C++
        # UEqn = fvm.ddt( U ) + fvm.div( phi, U ) + turbulence.divDevReff( U )

        UEqn = turbulence.divDevReff( U ) + ( ref.fvm.ddt( U ) + ref.fvm.div( phi, U ) )        

        UEqn.relax()

        if momentumPredictor :
            ref.solve( UEqn == -ref.fvc.grad( p ) )
            pass
           
        # --- PISO loop

        for corr in range( nCorr ) :
            rUA = 1.0 / UEqn.A()
            U << rUA * UEqn.H()

            phi << ( ref.fvc.interpolate(U) & mesh.Sf() ) + ref.fvc.ddtPhiCorr( rUA, U, phi )
         
            ref.adjustPhi( phi, U, p )
            
            # Non-orthogonal pressure corrector loop
            for nonOrth in range( nNonOrthCorr + 1 ):
                # Pressure corrector
                pEqn = ref.fvm.laplacian( rUA, p ) == ref.fvc.div( phi )

                pEqn.setReference( pRefCell, pRefValue )

                if corr == ( nCorr-1 ) and nonOrth == nNonOrthCorr :
                    pEqn.solve( mesh.solver( ref.word( "pFinal" ) ) ) 
                    pass
                else:
                    pEqn.solve()   
                    pass
                   
                if nonOrth == nNonOrthCorr:
                    phi -= pEqn.flux()
                    pass
                
                pass
            cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )       

            U -= rUA * ref.fvc.grad( p ) 
            U.correctBoundaryConditions()
            pass

        turbulence.correct()

        runTime.write()

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" << \
              "  ClockTime = " << runTime.elapsedClockTime() << " s" << ref.nl << ref.nl
        
        pass

    ref.ext_Info() << "End\n" << ref.nl 
    
    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
from Foam import FOAM_VERSION
if FOAM_VERSION( ">=", "020000" ):
   if __name__ == "__main__" :
      import sys, os
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   ref.ext_Info() << "\n\n To use this solver it is necessary to SWIG OpenFOAM-2.0.0 or higher\n"
   pass


#--------------------------------------------------------------------------------------
