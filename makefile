# Minimal makefile using PETSc's build system

APP := PETSc
SRC := PETSc.c
OBJ := $(SRC:.c=.o)

# Pull in PETSc's build settings (paths, libs, rules)
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Build the executable and link with PETSc-provided libs
$(APP):
	${CLINKER} -o $(APP) $(OBJ) ${PETSC_LIB}
	${RM} $(OBJ)