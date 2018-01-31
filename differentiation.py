import numpy as np
import dolfin


def eval_grad_foo(foo):
    '''Exact derivatives'''

    def evaluate(x):
        x = np.fromiter(x, dtype=float)

        V = foo.function_space()
        el = V.element()
        mesh = V.mesh()
        # Find the cell with point
        x_point = dolfin.Point(*x) 
        cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = dolfin.Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        shape = dolfin.grad(foo).ufl_shape
        # grad grad grad ... ntime is stored in
        nslots = np.prod(shape)

        # Array for values with derivatives of all basis functions. 4 * element dim
        values = np.zeros(nslots*el.space_dimension(), dtype=float)
        # Compute all 2nd order derivatives
        el.evaluate_basis_derivatives_all(1, values, x, coordinate_dofs, cell.orientation())
        # Reshape such that colums are [d/dxx, d/dxy, d/dyx, d/dyy]
        values = values.reshape((-1, nslots))

        # Get expansion coefs on cell. Alternative to this is f.restrict(...)
        dofs = V.dofmap().cell_dofs(cell_id)
        dofs = foo.vector()[dofs]

        # Perform reduction on each colum - you get that deriv of foo
        values = np.array([np.inner(row, dofs) for row in values.T])
        values = values.reshape(shape)

        return values
    return evaluate
