import odl
import myOperators
X = odl.uniform_discr(0, 128, 128) ** 2
A = myOperators.RealFourierTransform(X)
x = odl.phantom.white_noise(A.domain)
y = odl.phantom.white_noise(A.range)
t1 = A(x).inner(y)
t2 = x.inner(A.adjoint(y))
print(t1 / t2)

X = odl.rn((8, 8))
Y = odl.rn((2, 2))
S = myOperators.Subsampling(X, Y)
x = X.one()
y = S(x)


def deform_param(x, param):
    embedding = ops.Embedding_Affine(
        param.space, x.space.tangent_bundle)

    deform_op = odl.deform.LinDeformFixedDisp(embedding(param))

    return deform_op(x)
