       �K"	  @�t�Abrain.Event:2�v�>�i     4Nd	$l�t�A"��
f
S/sPlaceholder*
shape:���������P*
dtype0*'
_output_shapes
:���������P
f
R/rPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
h
S_/s_Placeholder*
shape:���������P*
dtype0*'
_output_shapes
:���������P
�
8Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
�
7Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
�
9Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *���>*+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
�
GActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes

:P*

seed
�
6Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulGActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal9Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
2Actor/eval_net/l1/kernel/Initializer/random_normalAdd6Actor/eval_net/l1/kernel/Initializer/random_normal/mul7Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
Actor/eval_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P
�
Actor/eval_net/l1/kernel/AssignAssignActor/eval_net/l1/kernel2Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
Actor/eval_net/l1/kernel/readIdentityActor/eval_net/l1/kernel*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
(Actor/eval_net/l1/bias/Initializer/ConstConst*
valueB*���=*)
_class
loc:@Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
�
Actor/eval_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:
�
Actor/eval_net/l1/bias/AssignAssignActor/eval_net/l1/bias(Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
Actor/eval_net/l1/bias/readIdentityActor/eval_net/l1/bias*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
�
Actor/eval_net/l1/MatMulMatMulS/sActor/eval_net/l1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
Actor/eval_net/l1/BiasAddBiasAddActor/eval_net/l1/MatMulActor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
k
Actor/eval_net/l1/ReluReluActor/eval_net/l1/BiasAdd*'
_output_shapes
:���������*
T0
�
9Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
�
8Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
�
:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *���>*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0
�
HActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
7Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulHActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
3Actor/eval_net/a/a/kernel/Initializer/random_normalAdd7Actor/eval_net/a/a/kernel/Initializer/random_normal/mul8Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
Actor/eval_net/a/a/kernel
VariableV2*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
 Actor/eval_net/a/a/kernel/AssignAssignActor/eval_net/a/a/kernel3Actor/eval_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(
�
Actor/eval_net/a/a/kernel/readIdentityActor/eval_net/a/a/kernel*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
)Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*���=**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
�
Actor/eval_net/a/a/bias
VariableV2**
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Actor/eval_net/a/a/bias/AssignAssignActor/eval_net/a/a/bias)Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
�
Actor/eval_net/a/a/bias/readIdentityActor/eval_net/a/a/bias*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
�
Actor/eval_net/a/a/MatMulMatMulActor/eval_net/l1/ReluActor/eval_net/a/a/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
Actor/eval_net/a/a/BiasAddBiasAddActor/eval_net/a/a/MatMulActor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
m
Actor/eval_net/a/a/TanhTanhActor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:���������
`
Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
�
Actor/eval_net/a/scaled_aMulActor/eval_net/a/a/TanhActor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:���������
�
:Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"P      *-
_class#
!loc:@Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
�
9Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@Actor/target_net/l1/kernel*
dtype0
�
;Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���>*-
_class#
!loc:@Actor/target_net/l1/kernel
�
IActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:Actor/target_net/l1/kernel/Initializer/random_normal/shape*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
seed2(*
dtype0*
_output_shapes

:P*

seed
�
8Actor/target_net/l1/kernel/Initializer/random_normal/mulMulIActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;Actor/target_net/l1/kernel/Initializer/random_normal/stddev*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P*
T0
�
4Actor/target_net/l1/kernel/Initializer/random_normalAdd8Actor/target_net/l1/kernel/Initializer/random_normal/mul9Actor/target_net/l1/kernel/Initializer/random_normal/mean*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P*
T0
�
Actor/target_net/l1/kernel
VariableV2*
shape
:P*
dtype0*
_output_shapes

:P*
shared_name *-
_class#
!loc:@Actor/target_net/l1/kernel*
	container 
�
!Actor/target_net/l1/kernel/AssignAssignActor/target_net/l1/kernel4Actor/target_net/l1/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel
�
Actor/target_net/l1/kernel/readIdentityActor/target_net/l1/kernel*
_output_shapes

:P*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel
�
*Actor/target_net/l1/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*���=*+
_class!
loc:@Actor/target_net/l1/bias*
dtype0
�
Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@Actor/target_net/l1/bias*
	container *
shape:
�
Actor/target_net/l1/bias/AssignAssignActor/target_net/l1/bias*Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
�
Actor/target_net/l1/bias/readIdentityActor/target_net/l1/bias*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
_output_shapes
:
�
Actor/target_net/l1/MatMulMatMulS_/s_Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Actor/target_net/l1/BiasAddBiasAddActor/target_net/l1/MatMulActor/target_net/l1/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
o
Actor/target_net/l1/ReluReluActor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:���������
�
;Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *.
_class$
" loc:@Actor/target_net/a/a/kernel
�
:Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
�
<Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *���>*.
_class$
" loc:@Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
�
JActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
seed28
�
9Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulJActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel
�
5Actor/target_net/a/a/kernel/Initializer/random_normalAdd9Actor/target_net/a/a/kernel/Initializer/random_normal/mul:Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
�
Actor/target_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@Actor/target_net/a/a/kernel*
	container *
shape
:
�
"Actor/target_net/a/a/kernel/AssignAssignActor/target_net/a/a/kernel5Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
�
 Actor/target_net/a/a/kernel/readIdentityActor/target_net/a/a/kernel*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
�
+Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*���=*,
_class"
 loc:@Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
�
Actor/target_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@Actor/target_net/a/a/bias*
	container *
shape:
�
 Actor/target_net/a/a/bias/AssignAssignActor/target_net/a/a/bias+Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
�
Actor/target_net/a/a/bias/readIdentityActor/target_net/a/a/bias*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
_output_shapes
:
�
Actor/target_net/a/a/MatMulMatMulActor/target_net/l1/Relu Actor/target_net/a/a/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
Actor/target_net/a/a/BiasAddBiasAddActor/target_net/a/a/MatMulActor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
q
Actor/target_net/a/a/TanhTanhActor/target_net/a/a/BiasAdd*'
_output_shapes
:���������*
T0
b
Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
�
Actor/target_net/a/scaled_aMulActor/target_net/a/a/TanhActor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:���������
J
mul/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
[
mulMulmul/xActor/target_net/l1/kernel/read*
T0*
_output_shapes

:P
L
mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
]
mul_1Mulmul_1/xActor/eval_net/l1/kernel/read*
T0*
_output_shapes

:P
?
addAddmulmul_1*
T0*
_output_shapes

:P
�
AssignAssignActor/target_net/l1/kerneladd*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(
L
mul_2/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
Y
mul_2Mulmul_2/xActor/target_net/l1/bias/read*
_output_shapes
:*
T0
L
mul_3/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
W
mul_3Mulmul_3/xActor/eval_net/l1/bias/read*
_output_shapes
:*
T0
?
add_1Addmul_2mul_3*
T0*
_output_shapes
:
�
Assign_1AssignActor/target_net/l1/biasadd_1*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
L
mul_4/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
`
mul_4Mulmul_4/x Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:
L
mul_5/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
^
mul_5Mulmul_5/xActor/eval_net/a/a/kernel/read*
_output_shapes

:*
T0
C
add_2Addmul_4mul_5*
_output_shapes

:*
T0
�
Assign_2AssignActor/target_net/a/a/kerneladd_2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel
L
mul_6/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
Z
mul_6Mulmul_6/xActor/target_net/a/a/bias/read*
T0*
_output_shapes
:
L
mul_7/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
X
mul_7Mulmul_7/xActor/eval_net/a/a/bias/read*
_output_shapes
:*
T0
?
add_3Addmul_6mul_7*
T0*
_output_shapes
:
�
Assign_3AssignActor/target_net/a/a/biasadd_3*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
p
Critic/StopGradientStopGradientActor/eval_net/a/scaled_a*
T0*'
_output_shapes
:���������
�
7Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
�
6Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
�
8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *���=**
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
�
FCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
seed2c*
dtype0*
_output_shapes

:P*

seed
�
5Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
1Critic/eval_net/l1/w1_s/Initializer/random_normalAdd5Critic/eval_net/l1/w1_s/Initializer/random_normal/mul6Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
�
Critic/eval_net/l1/w1_s
VariableV2**
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
�
Critic/eval_net/l1/w1_s/AssignAssignCritic/eval_net/l1/w1_s1Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
�
Critic/eval_net/l1/w1_s/readIdentityCritic/eval_net/l1/w1_s*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
7Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
�
6Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
�
8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *���=**
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
�
FCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
seed2l*
dtype0*
_output_shapes

:*

seed*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
�
5Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
�
1Critic/eval_net/l1/w1_a/Initializer/random_normalAdd5Critic/eval_net/l1/w1_a/Initializer/random_normal/mul6Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
�
Critic/eval_net/l1/w1_a
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container 
�
Critic/eval_net/l1/w1_a/AssignAssignCritic/eval_net/l1/w1_a1Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
�
Critic/eval_net/l1/w1_a/readIdentityCritic/eval_net/l1/w1_a*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
�
'Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*���=*(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
�
Critic/eval_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:
�
Critic/eval_net/l1/b1/AssignAssignCritic/eval_net/l1/b1'Critic/eval_net/l1/b1/Initializer/Const*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
Critic/eval_net/l1/b1/readIdentityCritic/eval_net/l1/b1*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:*
T0
�
Critic/eval_net/l1/MatMulMatMulS/sCritic/eval_net/l1/w1_s/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Critic/eval_net/l1/MatMul_1MatMulCritic/StopGradientCritic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Critic/eval_net/l1/addAddCritic/eval_net/l1/MatMulCritic/eval_net/l1/MatMul_1*'
_output_shapes
:���������*
T0
�
Critic/eval_net/l1/add_1AddCritic/eval_net/l1/addCritic/eval_net/l1/b1/read*'
_output_shapes
:���������*
T0
k
Critic/eval_net/l1/ReluReluCritic/eval_net/l1/add_1*
T0*'
_output_shapes
:���������
�
>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
�
=Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
�
?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *���=*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
�
MCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
seed2~*
dtype0*
_output_shapes

:*

seed*
T0
�
<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulMCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormal?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
8Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul=Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
Critic/eval_net/q/dense/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container 
�
%Critic/eval_net/q/dense/kernel/AssignAssignCritic/eval_net/q/dense/kernel8Critic/eval_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
�
#Critic/eval_net/q/dense/kernel/readIdentityCritic/eval_net/q/dense/kernel*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
.Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*���=*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
�
Critic/eval_net/q/dense/bias
VariableV2*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0
�
#Critic/eval_net/q/dense/bias/AssignAssignCritic/eval_net/q/dense/bias.Critic/eval_net/q/dense/bias/Initializer/Const*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
!Critic/eval_net/q/dense/bias/readIdentityCritic/eval_net/q/dense/bias*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
�
Critic/eval_net/q/dense/MatMulMatMulCritic/eval_net/l1/Relu#Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Critic/eval_net/q/dense/BiasAddBiasAddCritic/eval_net/q/dense/MatMul!Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
9Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"P      *,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
�
8Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
�
:Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*,
_class"
 loc:@Critic/target_net/l1/w1_s*
dtype0
�
HCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_s/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
seed2�*
dtype0*
_output_shapes

:P
�
7Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
�
3Critic/target_net/l1/w1_s/Initializer/random_normalAdd7Critic/target_net/l1/w1_s/Initializer/random_normal/mul8Critic/target_net/l1/w1_s/Initializer/random_normal/mean*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P*
T0
�
Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *,
_class"
 loc:@Critic/target_net/l1/w1_s*
	container *
shape
:P
�
 Critic/target_net/l1/w1_s/AssignAssignCritic/target_net/l1/w1_s3Critic/target_net/l1/w1_s/Initializer/random_normal*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
�
Critic/target_net/l1/w1_s/readIdentityCritic/target_net/l1/w1_s*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
�
9Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
�
8Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
�
:Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *���=*,
_class"
 loc:@Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
�
HCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
seed2�*
dtype0*
_output_shapes

:
�
7Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:
�
3Critic/target_net/l1/w1_a/Initializer/random_normalAdd7Critic/target_net/l1/w1_a/Initializer/random_normal/mul8Critic/target_net/l1/w1_a/Initializer/random_normal/mean*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
�
Critic/target_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@Critic/target_net/l1/w1_a*
	container *
shape
:
�
 Critic/target_net/l1/w1_a/AssignAssignCritic/target_net/l1/w1_a3Critic/target_net/l1/w1_a/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(
�
Critic/target_net/l1/w1_a/readIdentityCritic/target_net/l1/w1_a*
_output_shapes

:*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a
�
)Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*���=**
_class 
loc:@Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
�
Critic/target_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/target_net/l1/b1
�
Critic/target_net/l1/b1/AssignAssignCritic/target_net/l1/b1)Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
Critic/target_net/l1/b1/readIdentityCritic/target_net/l1/b1*
T0**
_class 
loc:@Critic/target_net/l1/b1*
_output_shapes

:
�
Critic/target_net/l1/MatMulMatMulS_/s_Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Critic/target_net/l1/MatMul_1MatMulActor/target_net/a/scaled_aCritic/target_net/l1/w1_a/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Critic/target_net/l1/addAddCritic/target_net/l1/MatMulCritic/target_net/l1/MatMul_1*'
_output_shapes
:���������*
T0
�
Critic/target_net/l1/add_1AddCritic/target_net/l1/addCritic/target_net/l1/b1/read*
T0*'
_output_shapes
:���������
o
Critic/target_net/l1/ReluReluCritic/target_net/l1/add_1*'
_output_shapes
:���������*
T0
�
@Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
�
?Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
�
ACritic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
dtype0
�
OCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
seed2�
�
>Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulOCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalACritic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:
�
:Critic/target_net/q/dense/kernel/Initializer/random_normalAdd>Critic/target_net/q/dense/kernel/Initializer/random_normal/mul?Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel
�
 Critic/target_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
	container *
shape
:
�
'Critic/target_net/q/dense/kernel/AssignAssign Critic/target_net/q/dense/kernel:Critic/target_net/q/dense/kernel/Initializer/random_normal*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
%Critic/target_net/q/dense/kernel/readIdentity Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel
�
0Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*���=*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
�
Critic/target_net/q/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@Critic/target_net/q/dense/bias
�
%Critic/target_net/q/dense/bias/AssignAssignCritic/target_net/q/dense/bias0Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
�
#Critic/target_net/q/dense/bias/readIdentityCritic/target_net/q/dense/bias*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
_output_shapes
:
�
 Critic/target_net/q/dense/MatMulMatMulCritic/target_net/l1/Relu%Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
!Critic/target_net/q/dense/BiasAddBiasAdd Critic/target_net/q/dense/MatMul#Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
S
target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
x
target_q/mulMultarget_q/mul/x!Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:���������
X
target_q/addAddR/rtarget_q/mul*'
_output_shapes
:���������*
T0
�
TD_error/SquaredDifferenceSquaredDifferencetarget_q/addCritic/eval_net/q/dense/BiasAdd*'
_output_shapes
:���������*
T0
_
TD_error/ConstConst*
dtype0*
_output_shapes
:*
valueB"       

TD_error/MeanMeanTD_error/SquaredDifferenceTD_error/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Z
C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
C_train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
C_train/gradients/FillFillC_train/gradients/ShapeC_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
2C_train/gradients/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
,C_train/gradients/TD_error/Mean_grad/ReshapeReshapeC_train/gradients/Fill2C_train/gradients/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
*C_train/gradients/TD_error/Mean_grad/ShapeShapeTD_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
)C_train/gradients/TD_error/Mean_grad/TileTile,C_train/gradients/TD_error/Mean_grad/Reshape*C_train/gradients/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
,C_train/gradients/TD_error/Mean_grad/Shape_1ShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
o
,C_train/gradients/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
t
*C_train/gradients/TD_error/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
)C_train/gradients/TD_error/Mean_grad/ProdProd,C_train/gradients/TD_error/Mean_grad/Shape_1*C_train/gradients/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
v
,C_train/gradients/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
+C_train/gradients/TD_error/Mean_grad/Prod_1Prod,C_train/gradients/TD_error/Mean_grad/Shape_2,C_train/gradients/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
.C_train/gradients/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
,C_train/gradients/TD_error/Mean_grad/MaximumMaximum+C_train/gradients/TD_error/Mean_grad/Prod_1.C_train/gradients/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
-C_train/gradients/TD_error/Mean_grad/floordivFloorDiv)C_train/gradients/TD_error/Mean_grad/Prod,C_train/gradients/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
)C_train/gradients/TD_error/Mean_grad/CastCast-C_train/gradients/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
,C_train/gradients/TD_error/Mean_grad/truedivRealDiv)C_train/gradients/TD_error/Mean_grad/Tile)C_train/gradients/TD_error/Mean_grad/Cast*'
_output_shapes
:���������*
T0
�
7C_train/gradients/TD_error/SquaredDifference_grad/ShapeShapetarget_q/add*
T0*
out_type0*
_output_shapes
:
�
9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1ShapeCritic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs7C_train/gradients/TD_error/SquaredDifference_grad/Shape9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8C_train/gradients/TD_error/SquaredDifference_grad/scalarConst-^C_train/gradients/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
5C_train/gradients/TD_error/SquaredDifference_grad/MulMul8C_train/gradients/TD_error/SquaredDifference_grad/scalar,C_train/gradients/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
5C_train/gradients/TD_error/SquaredDifference_grad/subSubtarget_q/addCritic/eval_net/q/dense/BiasAdd-^C_train/gradients/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
7C_train/gradients/TD_error/SquaredDifference_grad/mul_1Mul5C_train/gradients/TD_error/SquaredDifference_grad/Mul5C_train/gradients/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
5C_train/gradients/TD_error/SquaredDifference_grad/SumSum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeReshape5C_train/gradients/TD_error/SquaredDifference_grad/Sum7C_train/gradients/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7C_train/gradients/TD_error/SquaredDifference_grad/Sum_1Sum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1IC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1Reshape7C_train/gradients/TD_error/SquaredDifference_grad/Sum_19C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
5C_train/gradients/TD_error/SquaredDifference_grad/NegNeg;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
BC_train/gradients/TD_error/SquaredDifference_grad/tuple/group_depsNoOp6^C_train/gradients/TD_error/SquaredDifference_grad/Neg:^C_train/gradients/TD_error/SquaredDifference_grad/Reshape
�
JC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@C_train/gradients/TD_error/SquaredDifference_grad/Reshape
�
LC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity5C_train/gradients/TD_error/SquaredDifference_grad/NegC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
�
BC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
�
GC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpC^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradM^C_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1
�
OC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1H^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
QC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradH^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*U
_classK
IGloc:@C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency#Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/ReluOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
FC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOp=^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul?^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
�
NC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulG^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
PC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1Identity>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1G^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
�
7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGradNC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyCritic/eval_net/l1/Relu*'
_output_shapes
:���������*
T0
�
5C_train/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
�
7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
EC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3C_train/gradients/Critic/eval_net/l1/add_1_grad/SumSum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradEC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape3C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradGC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_17C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
@C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape:^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
�
HC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeA^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:���������
�
JC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1A^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
�
3C_train/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
�
5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0
�
CC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs3C_train/gradients/Critic/eval_net/l1/add_grad/Shape5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1C_train/gradients/Critic/eval_net/l1/add_grad/SumSumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyCC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5C_train/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape1C_train/gradients/Critic/eval_net/l1/add_grad/Sum3C_train/gradients/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_1SumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyEC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_15C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
>C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp6^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape8^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1
�
FC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity5C_train/gradients/Critic/eval_net/l1/add_grad/Reshape?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape
�
HC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:���������
�
7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulMatMulFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyCritic/eval_net/l1/w1_s/read*'
_output_shapes
:���������P*
transpose_a( *
transpose_b(*
T0
�
9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:P*
transpose_a(
�
AC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul:^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1
�
IC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulB^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:���������P
�
KC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1B^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:P*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1
�
9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Critic/eval_net/l1/w1_a/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradientHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
CC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp:^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul<^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
�
KC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulD^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
MC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1D^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*N
_classD
B@loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
�
!C_train/beta1_power/initial_valueConst*
valueB
 *fff?*(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
�
C_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape: 
�
C_train/beta1_power/AssignAssignC_train/beta1_power!C_train/beta1_power/initial_value*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
C_train/beta1_power/readIdentityC_train/beta1_power*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
!C_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*(
_class
loc:@Critic/eval_net/l1/b1
�
C_train/beta2_power
VariableV2*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape: *
dtype0
�
C_train/beta2_power/AssignAssignC_train/beta2_power!C_train/beta2_power/initial_value*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
�
C_train/beta2_power/readIdentityC_train/beta2_power*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
FC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
�
<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillFC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes

:P
�
$C_train/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P
�
+C_train/Critic/eval_net/l1/w1_s/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_s/Adam6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
�
)C_train/Critic/eval_net/l1/w1_s/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_s/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
HC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
�
>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillHC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensor>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes

:P
�
&C_train/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
shape
:P*
dtype0*
_output_shapes

:P*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container 
�
-C_train/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_s/Adam_18C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
_output_shapes

:P*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(
�
+C_train/Critic/eval_net/l1/w1_s/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_s/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
�
$C_train/Critic/eval_net/l1/w1_a/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a
�
+C_train/Critic/eval_net/l1/w1_a/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_a/Adam6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
�
)C_train/Critic/eval_net/l1/w1_a/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_a/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
�
8C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
�
&C_train/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0
�
-C_train/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_a/Adam_18C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
�
+C_train/Critic/eval_net/l1/w1_a/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_a/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
�
4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
"C_train/Critic/eval_net/l1/b1/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1
�
)C_train/Critic/eval_net/l1/b1/Adam/AssignAssign"C_train/Critic/eval_net/l1/b1/Adam4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
'C_train/Critic/eval_net/l1/b1/Adam/readIdentity"C_train/Critic/eval_net/l1/b1/Adam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
�
6C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
$C_train/Critic/eval_net/l1/b1/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container 
�
+C_train/Critic/eval_net/l1/b1/Adam_1/AssignAssign$C_train/Critic/eval_net/l1/b1/Adam_16C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
)C_train/Critic/eval_net/l1/b1/Adam_1/readIdentity$C_train/Critic/eval_net/l1/b1/Adam_1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
�
=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0
�
+C_train/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container 
�
2C_train/Critic/eval_net/q/dense/kernel/Adam/AssignAssign+C_train/Critic/eval_net/q/dense/kernel/Adam=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
0C_train/Critic/eval_net/q/dense/kernel/Adam/readIdentity+C_train/Critic/eval_net/q/dense/kernel/Adam*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
�
?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
�
-C_train/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
4C_train/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign-C_train/Critic/eval_net/q/dense/kernel/Adam_1?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
2C_train/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity-C_train/Critic/eval_net/q/dense/kernel/Adam_1*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*    *
dtype0
�
)C_train/Critic/eval_net/q/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
�
0C_train/Critic/eval_net/q/dense/bias/Adam/AssignAssign)C_train/Critic/eval_net/q/dense/bias/Adam;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
�
.C_train/Critic/eval_net/q/dense/bias/Adam/readIdentity)C_train/Critic/eval_net/q/dense/bias/Adam*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
�
=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
�
+C_train/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container 
�
2C_train/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign+C_train/Critic/eval_net/q/dense/bias/Adam_1=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
�
0C_train/Critic/eval_net/q/dense/bias/Adam_1/readIdentity+C_train/Critic/eval_net/q/dense/bias/Adam_1*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
_
C_train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Y
C_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
5C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_s$C_train/Critic/eval_net/l1/w1_s/Adam&C_train/Critic/eval_net/l1/w1_s/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonKC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:P*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
�
5C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_a$C_train/Critic/eval_net/l1/w1_a/Adam&C_train/Critic/eval_net/l1/w1_a/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonMC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
�
3C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam	ApplyAdamCritic/eval_net/l1/b1"C_train/Critic/eval_net/l1/b1/Adam$C_train/Critic/eval_net/l1/b1/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonJC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
�
<C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/kernel+C_train/Critic/eval_net/q/dense/kernel/Adam-C_train/Critic/eval_net/q/dense/kernel/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonPC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
�
:C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/bias)C_train/Critic/eval_net/q/dense/bias/Adam+C_train/Critic/eval_net/q/dense/bias/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonQC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
C_train/Adam/mulMulC_train/beta1_power/readC_train/Adam/beta14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
C_train/Adam/AssignAssignC_train/beta1_powerC_train/Adam/mul*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
�
C_train/Adam/mul_1MulC_train/beta2_power/readC_train/Adam/beta24^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
C_train/Adam/Assign_1AssignC_train/beta2_powerC_train/Adam/mul_1*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
C_train/AdamNoOp^C_train/Adam/Assign^C_train/Adam/Assign_14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam
u
a_grad/gradients/ShapeShapeCritic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
_
a_grad/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
a_grad/gradients/FillFilla_grad/gradients/Shapea_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:���������
�
Aa_grad/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrada_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
�
;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMula_grad/gradients/Fill#Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
=a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/Relua_grad/gradients/Fill*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
�
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Da_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
2a_grad/gradients/Critic/eval_net/l1/add_1_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradDa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape2a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradFa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
8a_grad/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_16a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
2a_grad/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
�
4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
Ba_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
0a_grad/gradients/Critic/eval_net/l1/add_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeBa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
4a_grad/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape0a_grad/gradients/Critic/eval_net/l1/add_grad/Sum2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeDa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_14a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
8a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Critic/eval_net/l1/w1_a/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
:a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradient6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
L
mul_8/xConst*
_output_shapes
: *
valueB
 *�p}?*
dtype0
^
mul_8Mulmul_8/xCritic/target_net/l1/w1_s/read*
T0*
_output_shapes

:P
L
mul_9/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
\
mul_9Mulmul_9/xCritic/eval_net/l1/w1_s/read*
_output_shapes

:P*
T0
C
add_4Addmul_8mul_9*
T0*
_output_shapes

:P
�
Assign_4AssignCritic/target_net/l1/w1_sadd_4*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
M
mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *�p}?
`
mul_10Mulmul_10/xCritic/target_net/l1/w1_a/read*
T0*
_output_shapes

:
M
mul_11/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
^
mul_11Mulmul_11/xCritic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
E
add_5Addmul_10mul_11*
T0*
_output_shapes

:
�
Assign_5AssignCritic/target_net/l1/w1_aadd_5*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
M
mul_12/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
^
mul_12Mulmul_12/xCritic/target_net/l1/b1/read*
T0*
_output_shapes

:
M
mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
\
mul_13Mulmul_13/xCritic/eval_net/l1/b1/read*
T0*
_output_shapes

:
E
add_6Addmul_12mul_13*
T0*
_output_shapes

:
�
Assign_6AssignCritic/target_net/l1/b1add_6*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(
M
mul_14/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
g
mul_14Mulmul_14/x%Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
M
mul_15/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
e
mul_15Mulmul_15/x#Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
E
add_7Addmul_14mul_15*
T0*
_output_shapes

:
�
Assign_7Assign Critic/target_net/q/dense/kerneladd_7*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
M
mul_16/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
a
mul_16Mulmul_16/x#Critic/target_net/q/dense/bias/read*
_output_shapes
:*
T0
M
mul_17/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
_
mul_17Mulmul_17/x!Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
A
add_8Addmul_16mul_17*
T0*
_output_shapes
:
�
Assign_8AssignCritic/target_net/q/dense/biasadd_8*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
u
policy_grads/gradients/ShapeShapeActor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
e
 policy_grads/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
policy_grads/gradients/FillFillpolicy_grads/gradients/Shape policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:���������
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ShapeShapeActor/eval_net/a/a/Tanh*
_output_shapes
:*
T0*
out_type0
�
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Kpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulMulpolicy_grads/gradients/FillActor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:���������
�
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/SumSum9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulKpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ReshapeReshape9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1MulActor/eval_net/a/a/Tanhpolicy_grads/gradients/Fill*'
_output_shapes
:���������*
T0
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1Mpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradTanhGradActor/eval_net/a/a/Tanh=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:���������
�
Bpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradActor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMulActor/eval_net/l1/Relu<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradReluGrad<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulActor/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
Apolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
;policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMulMatMul;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradActor/eval_net/l1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������P*
transpose_a( 
�
=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
_output_shapes

:P*
transpose_a(*
transpose_b( *
T0
�
!A_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
�
A_train/beta1_power
VariableV2**
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A_train/beta1_power/AssignAssignA_train/beta1_power!A_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
�
A_train/beta1_power/readIdentityA_train/beta1_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
!A_train/beta2_power/initial_valueConst*
valueB
 *w�?**
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
�
A_train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias
�
A_train/beta2_power/AssignAssignA_train/beta2_power!A_train/beta2_power/initial_value**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
A_train/beta2_power/readIdentityA_train/beta2_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
GA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
�
=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillGA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensor=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:P
�
%A_train/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
,A_train/Actor/eval_net/l1/kernel/Adam/AssignAssign%A_train/Actor/eval_net/l1/kernel/Adam7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
*A_train/Actor/eval_net/l1/kernel/Adam/readIdentity%A_train/Actor/eval_net/l1/kernel/Adam*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
IA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0
�
?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    
�
9A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillIA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:P
�
'A_train/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
.A_train/Actor/eval_net/l1/kernel/Adam_1/AssignAssign'A_train/Actor/eval_net/l1/kernel/Adam_19A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
,A_train/Actor/eval_net/l1/kernel/Adam_1/readIdentity'A_train/Actor/eval_net/l1/kernel/Adam_1*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
#A_train/Actor/eval_net/l1/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias
�
*A_train/Actor/eval_net/l1/bias/Adam/AssignAssign#A_train/Actor/eval_net/l1/bias/Adam5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
(A_train/Actor/eval_net/l1/bias/Adam/readIdentity#A_train/Actor/eval_net/l1/bias/Adam*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:*
T0
�
7A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
%A_train/Actor/eval_net/l1/bias/Adam_1
VariableV2*)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
,A_train/Actor/eval_net/l1/bias/Adam_1/AssignAssign%A_train/Actor/eval_net/l1/bias/Adam_17A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
*A_train/Actor/eval_net/l1/bias/Adam_1/readIdentity%A_train/Actor/eval_net/l1/bias/Adam_1*
_output_shapes
:*
T0*)
_class
loc:@Actor/eval_net/l1/bias
�
8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
�
&A_train/Actor/eval_net/a/a/kernel/Adam
VariableV2*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
-A_train/Actor/eval_net/a/a/kernel/Adam/AssignAssign&A_train/Actor/eval_net/a/a/kernel/Adam8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
�
+A_train/Actor/eval_net/a/a/kernel/Adam/readIdentity&A_train/Actor/eval_net/a/a/kernel/Adam*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:*
T0
�
:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
�
(A_train/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
/A_train/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign(A_train/Actor/eval_net/a/a/kernel/Adam_1:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
�
-A_train/Actor/eval_net/a/a/kernel/Adam_1/readIdentity(A_train/Actor/eval_net/a/a/kernel/Adam_1*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
�
$A_train/Actor/eval_net/a/a/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:
�
+A_train/Actor/eval_net/a/a/bias/Adam/AssignAssign$A_train/Actor/eval_net/a/a/bias/Adam6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
)A_train/Actor/eval_net/a/a/bias/Adam/readIdentity$A_train/Actor/eval_net/a/a/bias/Adam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
�
8A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
�
&A_train/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias
�
-A_train/Actor/eval_net/a/a/bias/Adam_1/AssignAssign&A_train/Actor/eval_net/a/a/bias/Adam_18A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
�
+A_train/Actor/eval_net/a/a/bias/Adam_1/readIdentity&A_train/Actor/eval_net/a/a/bias/Adam_1*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
_
A_train/Adam/learning_rateConst*
valueB
 *o��*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
Y
A_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
6A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdamActor/eval_net/l1/kernel%A_train/Actor/eval_net/l1/kernel/Adam'A_train/Actor/eval_net/l1/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes

:P*
use_locking( *
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
4A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam	ApplyAdamActor/eval_net/l1/bias#A_train/Actor/eval_net/l1/bias/Adam%A_train/Actor/eval_net/l1/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonApolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*)
_class
loc:@Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:
�
7A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdamActor/eval_net/a/a/kernel&A_train/Actor/eval_net/a/a/kernel/Adam(A_train/Actor/eval_net/a/a/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
5A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdamActor/eval_net/a/a/bias$A_train/Actor/eval_net/a/a/bias/Adam&A_train/Actor/eval_net/a/a/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonBpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad**
_class 
loc:@Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
A_train/Adam/mulMulA_train/beta1_power/readA_train/Adam/beta16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@Actor/eval_net/a/a/bias
�
A_train/Adam/AssignAssignA_train/beta1_powerA_train/Adam/mul*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�
A_train/Adam/mul_1MulA_train/beta2_power/readA_train/Adam/beta26^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
A_train/Adam/Assign_1AssignA_train/beta2_powerA_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
�
A_train/AdamNoOp^A_train/Adam/Assign^A_train/Adam/Assign_16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam
�
initNoOp,^A_train/Actor/eval_net/a/a/bias/Adam/Assign.^A_train/Actor/eval_net/a/a/bias/Adam_1/Assign.^A_train/Actor/eval_net/a/a/kernel/Adam/Assign0^A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign+^A_train/Actor/eval_net/l1/bias/Adam/Assign-^A_train/Actor/eval_net/l1/bias/Adam_1/Assign-^A_train/Actor/eval_net/l1/kernel/Adam/Assign/^A_train/Actor/eval_net/l1/kernel/Adam_1/Assign^A_train/beta1_power/Assign^A_train/beta2_power/Assign^Actor/eval_net/a/a/bias/Assign!^Actor/eval_net/a/a/kernel/Assign^Actor/eval_net/l1/bias/Assign ^Actor/eval_net/l1/kernel/Assign!^Actor/target_net/a/a/bias/Assign#^Actor/target_net/a/a/kernel/Assign ^Actor/target_net/l1/bias/Assign"^Actor/target_net/l1/kernel/Assign*^C_train/Critic/eval_net/l1/b1/Adam/Assign,^C_train/Critic/eval_net/l1/b1/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_a/Adam/Assign.^C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_s/Adam/Assign.^C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign1^C_train/Critic/eval_net/q/dense/bias/Adam/Assign3^C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign3^C_train/Critic/eval_net/q/dense/kernel/Adam/Assign5^C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign^C_train/beta1_power/Assign^C_train/beta2_power/Assign^Critic/eval_net/l1/b1/Assign^Critic/eval_net/l1/w1_a/Assign^Critic/eval_net/l1/w1_s/Assign$^Critic/eval_net/q/dense/bias/Assign&^Critic/eval_net/q/dense/kernel/Assign^Critic/target_net/l1/b1/Assign!^Critic/target_net/l1/w1_a/Assign!^Critic/target_net/l1/w1_s/Assign&^Critic/target_net/q/dense/bias/Assign(^Critic/target_net/q/dense/kernel/Assign"&:�i���     e��S	3�m�t�AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
2
StopGradient

input"T
output"T"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.14.02v1.14.0-rc1-22-gaf24dc91b5��
f
S/sPlaceholder*
dtype0*'
_output_shapes
:���������P*
shape:���������P
f
R/rPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
h
S_/s_Placeholder*
dtype0*'
_output_shapes
:���������P*
shape:���������P
�
8Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
�
7Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
9Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*+
_class!
loc:@Actor/eval_net/l1/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
GActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:P*

seed*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
seed2
�
6Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulGActor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal9Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
2Actor/eval_net/l1/kernel/Initializer/random_normalAdd6Actor/eval_net/l1/kernel/Initializer/random_normal/mul7Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
Actor/eval_net/l1/kernel
VariableV2*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
Actor/eval_net/l1/kernel/AssignAssignActor/eval_net/l1/kernel2Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
Actor/eval_net/l1/kernel/readIdentityActor/eval_net/l1/kernel*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
(Actor/eval_net/l1/bias/Initializer/ConstConst*)
_class
loc:@Actor/eval_net/l1/bias*
valueB*���=*
dtype0*
_output_shapes
:
�
Actor/eval_net/l1/bias
VariableV2*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
Actor/eval_net/l1/bias/AssignAssignActor/eval_net/l1/bias(Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
Actor/eval_net/l1/bias/readIdentityActor/eval_net/l1/bias*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
�
Actor/eval_net/l1/MatMulMatMulS/sActor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
Actor/eval_net/l1/BiasAddBiasAddActor/eval_net/l1/MatMulActor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
k
Actor/eval_net/l1/ReluReluActor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:���������
�
9Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
8Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
HActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
7Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulHActor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal:Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
3Actor/eval_net/a/a/kernel/Initializer/random_normalAdd7Actor/eval_net/a/a/kernel/Initializer/random_normal/mul8Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
Actor/eval_net/a/a/kernel
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
 Actor/eval_net/a/a/kernel/AssignAssignActor/eval_net/a/a/kernel3Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
�
Actor/eval_net/a/a/kernel/readIdentityActor/eval_net/a/a/kernel*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
)Actor/eval_net/a/a/bias/Initializer/ConstConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB*���=*
dtype0*
_output_shapes
:
�
Actor/eval_net/a/a/bias
VariableV2*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0
�
Actor/eval_net/a/a/bias/AssignAssignActor/eval_net/a/a/bias)Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
�
Actor/eval_net/a/a/bias/readIdentityActor/eval_net/a/a/bias*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
�
Actor/eval_net/a/a/MatMulMatMulActor/eval_net/l1/ReluActor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
Actor/eval_net/a/a/BiasAddBiasAddActor/eval_net/a/a/MatMulActor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
m
Actor/eval_net/a/a/TanhTanhActor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:���������
`
Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
�
Actor/eval_net/a/scaled_aMulActor/eval_net/a/a/TanhActor/eval_net/a/scaled_a/y*'
_output_shapes
:���������*
T0
�
:Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB"P      *
dtype0*
_output_shapes
:
�
9Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB
 *    *
dtype0
�
;Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@Actor/target_net/l1/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
IActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:P*

seed*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
seed2(
�
8Actor/target_net/l1/kernel/Initializer/random_normal/mulMulIActor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes

:P*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel
�
4Actor/target_net/l1/kernel/Initializer/random_normalAdd8Actor/target_net/l1/kernel/Initializer/random_normal/mul9Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
_output_shapes

:P
�
Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:P*
shared_name *-
_class#
!loc:@Actor/target_net/l1/kernel*
	container *
shape
:P
�
!Actor/target_net/l1/kernel/AssignAssignActor/target_net/l1/kernel4Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
Actor/target_net/l1/kernel/readIdentityActor/target_net/l1/kernel*
_output_shapes

:P*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel
�
*Actor/target_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@Actor/target_net/l1/bias*
valueB*���=*
dtype0*
_output_shapes
:
�
Actor/target_net/l1/bias
VariableV2*+
_class!
loc:@Actor/target_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Actor/target_net/l1/bias/AssignAssignActor/target_net/l1/bias*Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
�
Actor/target_net/l1/bias/readIdentityActor/target_net/l1/bias*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
_output_shapes
:
�
Actor/target_net/l1/MatMulMatMulS_/s_Actor/target_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
Actor/target_net/l1/BiasAddBiasAddActor/target_net/l1/MatMulActor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
Actor/target_net/l1/ReluReluActor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:���������
�
;Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
:Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@Actor/target_net/a/a/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
JActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
seed28
�
9Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulJActor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
�
5Actor/target_net/a/a/kernel/Initializer/random_normalAdd9Actor/target_net/a/a/kernel/Initializer/random_normal/mul:Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
�
Actor/target_net/a/a/kernel
VariableV2*
shared_name *.
_class$
" loc:@Actor/target_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"Actor/target_net/a/a/kernel/AssignAssignActor/target_net/a/a/kernel5Actor/target_net/a/a/kernel/Initializer/random_normal*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
 Actor/target_net/a/a/kernel/readIdentityActor/target_net/a/a/kernel*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
_output_shapes

:
�
+Actor/target_net/a/a/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Actor/target_net/a/a/bias*
valueB*���=
�
Actor/target_net/a/a/bias
VariableV2*
shared_name *,
_class"
 loc:@Actor/target_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
 Actor/target_net/a/a/bias/AssignAssignActor/target_net/a/a/bias+Actor/target_net/a/a/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias
�
Actor/target_net/a/a/bias/readIdentityActor/target_net/a/a/bias*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
_output_shapes
:
�
Actor/target_net/a/a/MatMulMatMulActor/target_net/l1/Relu Actor/target_net/a/a/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
�
Actor/target_net/a/a/BiasAddBiasAddActor/target_net/a/a/MatMulActor/target_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
q
Actor/target_net/a/a/TanhTanhActor/target_net/a/a/BiasAdd*'
_output_shapes
:���������*
T0
b
Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 
�
Actor/target_net/a/scaled_aMulActor/target_net/a/a/TanhActor/target_net/a/scaled_a/y*'
_output_shapes
:���������*
T0
J
mul/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
[
mulMulmul/xActor/target_net/l1/kernel/read*
T0*
_output_shapes

:P
L
mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
]
mul_1Mulmul_1/xActor/eval_net/l1/kernel/read*
_output_shapes

:P*
T0
?
addAddmulmul_1*
_output_shapes

:P*
T0
�
AssignAssignActor/target_net/l1/kerneladd*
T0*-
_class#
!loc:@Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(
L
mul_2/xConst*
_output_shapes
: *
valueB
 *�p}?*
dtype0
Y
mul_2Mulmul_2/xActor/target_net/l1/bias/read*
T0*
_output_shapes
:
L
mul_3/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
W
mul_3Mulmul_3/xActor/eval_net/l1/bias/read*
T0*
_output_shapes
:
?
add_1Addmul_2mul_3*
T0*
_output_shapes
:
�
Assign_1AssignActor/target_net/l1/biasadd_1*
use_locking(*
T0*+
_class!
loc:@Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
L
mul_4/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
`
mul_4Mulmul_4/x Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:
L
mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
^
mul_5Mulmul_5/xActor/eval_net/a/a/kernel/read*
_output_shapes

:*
T0
C
add_2Addmul_4mul_5*
T0*
_output_shapes

:
�
Assign_2AssignActor/target_net/a/a/kerneladd_2*
T0*.
_class$
" loc:@Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
L
mul_6/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
Z
mul_6Mulmul_6/xActor/target_net/a/a/bias/read*
T0*
_output_shapes
:
L
mul_7/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
X
mul_7Mulmul_7/xActor/eval_net/a/a/bias/read*
_output_shapes
:*
T0
?
add_3Addmul_6mul_7*
T0*
_output_shapes
:
�
Assign_3AssignActor/target_net/a/a/biasadd_3*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@Actor/target_net/a/a/bias*
validate_shape(
p
Critic/StopGradientStopGradientActor/eval_net/a/scaled_a*'
_output_shapes
:���������*
T0
�
7Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB"P      *
dtype0*
_output_shapes
:
�
6Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst**
_class 
loc:@Critic/eval_net/l1/w1_s*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
FCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:P*

seed*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
seed2c
�
5Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
1Critic/eval_net/l1/w1_s/Initializer/random_normalAdd5Critic/eval_net/l1/w1_s/Initializer/random_normal/mul6Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:P*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s
�
Critic/eval_net/l1/w1_s
VariableV2**
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
�
Critic/eval_net/l1/w1_s/AssignAssignCritic/eval_net/l1/w1_s1Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
�
Critic/eval_net/l1/w1_s/readIdentityCritic/eval_net/l1/w1_s*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
7Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
�
6Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
_output_shapes
: **
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0
�
8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst**
_class 
loc:@Critic/eval_net/l1/w1_a*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
FCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
seed2l
�
5Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulFCritic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal8Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
�
1Critic/eval_net/l1/w1_a/Initializer/random_normalAdd5Critic/eval_net/l1/w1_a/Initializer/random_normal/mul6Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
�
Critic/eval_net/l1/w1_a
VariableV2*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
�
Critic/eval_net/l1/w1_a/AssignAssignCritic/eval_net/l1/w1_a1Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
�
Critic/eval_net/l1/w1_a/readIdentityCritic/eval_net/l1/w1_a*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
�
'Critic/eval_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*(
_class
loc:@Critic/eval_net/l1/b1*
valueB*���=*
dtype0
�
Critic/eval_net/l1/b1
VariableV2*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
Critic/eval_net/l1/b1/AssignAssignCritic/eval_net/l1/b1'Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
Critic/eval_net/l1/b1/readIdentityCritic/eval_net/l1/b1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
�
Critic/eval_net/l1/MatMulMatMulS/sCritic/eval_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
�
Critic/eval_net/l1/MatMul_1MatMulCritic/StopGradientCritic/eval_net/l1/w1_a/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
�
Critic/eval_net/l1/addAddCritic/eval_net/l1/MatMulCritic/eval_net/l1/MatMul_1*'
_output_shapes
:���������*
T0
�
Critic/eval_net/l1/add_1AddCritic/eval_net/l1/addCritic/eval_net/l1/b1/read*
T0*'
_output_shapes
:���������
k
Critic/eval_net/l1/ReluReluCritic/eval_net/l1/add_1*'
_output_shapes
:���������*
T0
�
>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
=Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
MCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal>Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
seed2~
�
<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulMCritic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormal?Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
8Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd<Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul=Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
Critic/eval_net/q/dense/kernel
VariableV2*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
%Critic/eval_net/q/dense/kernel/AssignAssignCritic/eval_net/q/dense/kernel8Critic/eval_net/q/dense/kernel/Initializer/random_normal*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
#Critic/eval_net/q/dense/kernel/readIdentityCritic/eval_net/q/dense/kernel*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
.Critic/eval_net/q/dense/bias/Initializer/ConstConst*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
valueB*���=*
dtype0*
_output_shapes
:
�
Critic/eval_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
�
#Critic/eval_net/q/dense/bias/AssignAssignCritic/eval_net/q/dense/bias.Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
�
!Critic/eval_net/q/dense/bias/readIdentityCritic/eval_net/q/dense/bias*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
�
Critic/eval_net/q/dense/MatMulMatMulCritic/eval_net/l1/Relu#Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
�
Critic/eval_net/q/dense/BiasAddBiasAddCritic/eval_net/q/dense/MatMul!Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
9Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB"P      
�
8Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
_output_shapes
: *,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0
�
:Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@Critic/target_net/l1/w1_s*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
HCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
seed2�*
dtype0*
_output_shapes

:P*

seed
�
7Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P
�
3Critic/target_net/l1/w1_s/Initializer/random_normalAdd7Critic/target_net/l1/w1_s/Initializer/random_normal/mul8Critic/target_net/l1/w1_s/Initializer/random_normal/mean*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P*
T0
�
Critic/target_net/l1/w1_s
VariableV2*,
_class"
 loc:@Critic/target_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
�
 Critic/target_net/l1/w1_s/AssignAssignCritic/target_net/l1/w1_s3Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
�
Critic/target_net/l1/w1_s/readIdentityCritic/target_net/l1/w1_s*,
_class"
 loc:@Critic/target_net/l1/w1_s*
_output_shapes

:P*
T0
�
9Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB"      *
dtype0
�
8Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*,
_class"
 loc:@Critic/target_net/l1/w1_a*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
HCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
seed2�
�
7Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulHCritic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
�
3Critic/target_net/l1/w1_a/Initializer/random_normalAdd7Critic/target_net/l1/w1_a/Initializer/random_normal/mul8Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:
�
Critic/target_net/l1/w1_a
VariableV2*
shared_name *,
_class"
 loc:@Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
�
 Critic/target_net/l1/w1_a/AssignAssignCritic/target_net/l1/w1_a3Critic/target_net/l1/w1_a/Initializer/random_normal*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
�
Critic/target_net/l1/w1_a/readIdentityCritic/target_net/l1/w1_a*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
_output_shapes

:
�
)Critic/target_net/l1/b1/Initializer/ConstConst**
_class 
loc:@Critic/target_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
Critic/target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/target_net/l1/b1*
	container *
shape
:
�
Critic/target_net/l1/b1/AssignAssignCritic/target_net/l1/b1)Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
Critic/target_net/l1/b1/readIdentityCritic/target_net/l1/b1*
_output_shapes

:*
T0**
_class 
loc:@Critic/target_net/l1/b1
�
Critic/target_net/l1/MatMulMatMulS_/s_Critic/target_net/l1/w1_s/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
�
Critic/target_net/l1/MatMul_1MatMulActor/target_net/a/scaled_aCritic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
Critic/target_net/l1/addAddCritic/target_net/l1/MatMulCritic/target_net/l1/MatMul_1*'
_output_shapes
:���������*
T0
�
Critic/target_net/l1/add_1AddCritic/target_net/l1/addCritic/target_net/l1/b1/read*
T0*'
_output_shapes
:���������
o
Critic/target_net/l1/ReluReluCritic/target_net/l1/add_1*'
_output_shapes
:���������*
T0
�
@Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
?Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
ACritic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
OCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
seed2�
�
>Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulOCritic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalACritic/target_net/q/dense/kernel/Initializer/random_normal/stddev*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
�
:Critic/target_net/q/dense/kernel/Initializer/random_normalAdd>Critic/target_net/q/dense/kernel/Initializer/random_normal/mul?Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
�
 Critic/target_net/q/dense/kernel
VariableV2*
shared_name *3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
'Critic/target_net/q/dense/kernel/AssignAssign Critic/target_net/q/dense/kernel:Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
�
%Critic/target_net/q/dense/kernel/readIdentity Critic/target_net/q/dense/kernel*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
�
0Critic/target_net/q/dense/bias/Initializer/ConstConst*
_output_shapes
:*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
valueB*���=*
dtype0
�
Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@Critic/target_net/q/dense/bias*
	container *
shape:
�
%Critic/target_net/q/dense/bias/AssignAssignCritic/target_net/q/dense/bias0Critic/target_net/q/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias
�
#Critic/target_net/q/dense/bias/readIdentityCritic/target_net/q/dense/bias*
_output_shapes
:*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias
�
 Critic/target_net/q/dense/MatMulMatMulCritic/target_net/l1/Relu%Critic/target_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
!Critic/target_net/q/dense/BiasAddBiasAdd Critic/target_net/q/dense/MatMul#Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
S
target_q/mul/xConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
x
target_q/mulMultarget_q/mul/x!Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:���������
X
target_q/addAddR/rtarget_q/mul*
T0*'
_output_shapes
:���������
�
TD_error/SquaredDifferenceSquaredDifferencetarget_q/addCritic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:���������
_
TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

TD_error/MeanMeanTD_error/SquaredDifferenceTD_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
C_train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
C_train/gradients/FillFillC_train/gradients/ShapeC_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
2C_train/gradients/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
,C_train/gradients/TD_error/Mean_grad/ReshapeReshapeC_train/gradients/Fill2C_train/gradients/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
*C_train/gradients/TD_error/Mean_grad/ShapeShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
)C_train/gradients/TD_error/Mean_grad/TileTile,C_train/gradients/TD_error/Mean_grad/Reshape*C_train/gradients/TD_error/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
�
,C_train/gradients/TD_error/Mean_grad/Shape_1ShapeTD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
o
,C_train/gradients/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
t
*C_train/gradients/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
)C_train/gradients/TD_error/Mean_grad/ProdProd,C_train/gradients/TD_error/Mean_grad/Shape_1*C_train/gradients/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
v
,C_train/gradients/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
+C_train/gradients/TD_error/Mean_grad/Prod_1Prod,C_train/gradients/TD_error/Mean_grad/Shape_2,C_train/gradients/TD_error/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
p
.C_train/gradients/TD_error/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
,C_train/gradients/TD_error/Mean_grad/MaximumMaximum+C_train/gradients/TD_error/Mean_grad/Prod_1.C_train/gradients/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
-C_train/gradients/TD_error/Mean_grad/floordivFloorDiv)C_train/gradients/TD_error/Mean_grad/Prod,C_train/gradients/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
)C_train/gradients/TD_error/Mean_grad/CastCast-C_train/gradients/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
�
,C_train/gradients/TD_error/Mean_grad/truedivRealDiv)C_train/gradients/TD_error/Mean_grad/Tile)C_train/gradients/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
7C_train/gradients/TD_error/SquaredDifference_grad/ShapeShapetarget_q/add*
_output_shapes
:*
T0*
out_type0
�
9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1ShapeCritic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs7C_train/gradients/TD_error/SquaredDifference_grad/Shape9C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8C_train/gradients/TD_error/SquaredDifference_grad/scalarConst-^C_train/gradients/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
5C_train/gradients/TD_error/SquaredDifference_grad/MulMul8C_train/gradients/TD_error/SquaredDifference_grad/scalar,C_train/gradients/TD_error/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
5C_train/gradients/TD_error/SquaredDifference_grad/subSubtarget_q/addCritic/eval_net/q/dense/BiasAdd-^C_train/gradients/TD_error/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
7C_train/gradients/TD_error/SquaredDifference_grad/mul_1Mul5C_train/gradients/TD_error/SquaredDifference_grad/Mul5C_train/gradients/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
5C_train/gradients/TD_error/SquaredDifference_grad/SumSum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1GC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeReshape5C_train/gradients/TD_error/SquaredDifference_grad/Sum7C_train/gradients/TD_error/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
7C_train/gradients/TD_error/SquaredDifference_grad/Sum_1Sum7C_train/gradients/TD_error/SquaredDifference_grad/mul_1IC_train/gradients/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1Reshape7C_train/gradients/TD_error/SquaredDifference_grad/Sum_19C_train/gradients/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
5C_train/gradients/TD_error/SquaredDifference_grad/NegNeg;C_train/gradients/TD_error/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
BC_train/gradients/TD_error/SquaredDifference_grad/tuple/group_depsNoOp6^C_train/gradients/TD_error/SquaredDifference_grad/Neg:^C_train/gradients/TD_error/SquaredDifference_grad/Reshape
�
JC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity9C_train/gradients/TD_error/SquaredDifference_grad/ReshapeC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
LC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity5C_train/gradients/TD_error/SquaredDifference_grad/NegC^C_train/gradients/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
BC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
�
GC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpC^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradM^C_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1
�
OC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityLC_train/gradients/TD_error/SquaredDifference_grad/tuple/control_dependency_1H^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@C_train/gradients/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
QC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradH^C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@C_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency#Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b(*
T0
�
>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/ReluOC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
�
FC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOp=^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul?^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
�
NC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity<C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulG^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
PC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1Identity>C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1G^C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@C_train/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
�
7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGradNC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
5C_train/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
�
7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
EC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape7C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
3C_train/gradients/Critic/eval_net/l1/add_1_grad/SumSum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradEC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape3C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum5C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum7C_train/gradients/Critic/eval_net/l1/Relu_grad/ReluGradGC_train/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape5C_train/gradients/Critic/eval_net/l1/add_1_grad/Sum_17C_train/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0
�
@C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape:^C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1
�
HC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/add_1_grad/ReshapeA^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:���������
�
JC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1A^C_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/group_deps*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:*
T0
�
3C_train/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
out_type0*
_output_shapes
:*
T0
�
5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
CC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs3C_train/gradients/Critic/eval_net/l1/add_grad/Shape5C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
1C_train/gradients/Critic/eval_net/l1/add_grad/SumSumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyCC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5C_train/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape1C_train/gradients/Critic/eval_net/l1/add_grad/Sum3C_train/gradients/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_1SumHC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyEC_train/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape3C_train/gradients/Critic/eval_net/l1/add_grad/Sum_15C_train/gradients/Critic/eval_net/l1/add_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
>C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp6^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape8^C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1
�
FC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity5C_train/gradients/Critic/eval_net/l1/add_grad/Reshape?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape
�
HC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity7C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1?^C_train/gradients/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:���������
�
7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulMatMulFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependencyCritic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:���������P*
transpose_b(
�
9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sFC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:P*
transpose_b( *
T0
�
AC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp8^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul:^C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1
�
IC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity7C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMulB^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������P*
T0*J
_class@
><loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul
�
KC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity9C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1B^C_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:P
�
9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b(*
T0
�
;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradientHC_train/gradients/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
�
CC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp:^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul<^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
�
KC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity9C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulD^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*L
_classB
@>loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:���������*
T0
�
MC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity;C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1D^C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@C_train/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
!C_train/beta1_power/initial_valueConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
C_train/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container 
�
C_train/beta1_power/AssignAssignC_train/beta1_power!C_train/beta1_power/initial_value*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
�
C_train/beta1_power/readIdentityC_train/beta1_power*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: *
T0
�
!C_train/beta2_power/initial_valueConst*(
_class
loc:@Critic/eval_net/l1/b1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape: 
�
C_train/beta2_power/AssignAssignC_train/beta2_power!C_train/beta2_power/initial_value*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
C_train/beta2_power/readIdentityC_train/beta2_power*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
FC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0
�
<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
�
6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillFC_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor<C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
$C_train/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
+C_train/Critic/eval_net/l1/w1_s/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_s/Adam6C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
�
)C_train/Critic/eval_net/l1/w1_s/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_s/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
HC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"P      **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
�
>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
�
8C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillHC_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensor>C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
&C_train/Critic/eval_net/l1/w1_s/Adam_1
VariableV2**
_class 
loc:@Critic/eval_net/l1/w1_s*
	container *
shape
:P*
dtype0*
_output_shapes

:P*
shared_name 
�
-C_train/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_s/Adam_18C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
�
+C_train/Critic/eval_net/l1/w1_s/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_s/Adam_1*
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
_output_shapes

:P
�
6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
�
$C_train/Critic/eval_net/l1/w1_a/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:
�
+C_train/Critic/eval_net/l1/w1_a/Adam/AssignAssign$C_train/Critic/eval_net/l1/w1_a/Adam6C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
�
)C_train/Critic/eval_net/l1/w1_a/Adam/readIdentity$C_train/Critic/eval_net/l1/w1_a/Adam*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
_output_shapes

:
�
8C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    **
_class 
loc:@Critic/eval_net/l1/w1_a
�
&C_train/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@Critic/eval_net/l1/w1_a*
	container *
shape
:
�
-C_train/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign&C_train/Critic/eval_net/l1/w1_a/Adam_18C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
�
+C_train/Critic/eval_net/l1/w1_a/Adam_1/readIdentity&C_train/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0**
_class 
loc:@Critic/eval_net/l1/w1_a
�
4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
valueB*    *(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
�
"C_train/Critic/eval_net/l1/b1/Adam
VariableV2*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
)C_train/Critic/eval_net/l1/b1/Adam/AssignAssign"C_train/Critic/eval_net/l1/b1/Adam4C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
'C_train/Critic/eval_net/l1/b1/Adam/readIdentity"C_train/Critic/eval_net/l1/b1/Adam*
_output_shapes

:*
T0*(
_class
loc:@Critic/eval_net/l1/b1
�
6C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
valueB*    *(
_class
loc:@Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
�
$C_train/Critic/eval_net/l1/b1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@Critic/eval_net/l1/b1*
	container *
shape
:
�
+C_train/Critic/eval_net/l1/b1/Adam_1/AssignAssign$C_train/Critic/eval_net/l1/b1/Adam_16C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
)C_train/Critic/eval_net/l1/b1/Adam_1/readIdentity$C_train/Critic/eval_net/l1/b1/Adam_1*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes

:
�
=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
�
+C_train/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
2C_train/Critic/eval_net/q/dense/kernel/Adam/AssignAssign+C_train/Critic/eval_net/q/dense/kernel/Adam=C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
�
0C_train/Critic/eval_net/q/dense/kernel/Adam/readIdentity+C_train/Critic/eval_net/q/dense/kernel/Adam*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
�
-C_train/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
�
4C_train/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign-C_train/Critic/eval_net/q/dense/kernel/Adam_1?C_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
2C_train/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity-C_train/Critic/eval_net/q/dense/kernel/Adam_1*
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel*
_output_shapes

:
�
;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0
�
)C_train/Critic/eval_net/q/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
�
0C_train/Critic/eval_net/q/dense/bias/Adam/AssignAssign)C_train/Critic/eval_net/q/dense/bias/Adam;C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
�
.C_train/Critic/eval_net/q/dense/bias/Adam/readIdentity)C_train/Critic/eval_net/q/dense/bias/Adam*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
_output_shapes
:
�
=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
�
+C_train/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name */
_class%
#!loc:@Critic/eval_net/q/dense/bias*
	container *
shape:
�
2C_train/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign+C_train/Critic/eval_net/q/dense/bias/Adam_1=C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
0C_train/Critic/eval_net/q/dense/bias/Adam_1/readIdentity+C_train/Critic/eval_net/q/dense/bias/Adam_1*
_output_shapes
:*
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias
_
C_train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
C_train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Y
C_train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
5C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_s$C_train/Critic/eval_net/l1/w1_s/Adam&C_train/Critic/eval_net/l1/w1_s/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonKC_train/gradients/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:P
�
5C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdamCritic/eval_net/l1/w1_a$C_train/Critic/eval_net/l1/w1_a/Adam&C_train/Critic/eval_net/l1/w1_a/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonMC_train/gradients/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
�
3C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam	ApplyAdamCritic/eval_net/l1/b1"C_train/Critic/eval_net/l1/b1/Adam$C_train/Critic/eval_net/l1/b1/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonJC_train/gradients/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*(
_class
loc:@Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
�
<C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/kernel+C_train/Critic/eval_net/q/dense/kernel/Adam-C_train/Critic/eval_net/q/dense/kernel/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonPC_train/gradients/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*1
_class'
%#loc:@Critic/eval_net/q/dense/kernel
�
:C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdamCritic/eval_net/q/dense/bias)C_train/Critic/eval_net/q/dense/bias/Adam+C_train/Critic/eval_net/q/dense/bias/Adam_1C_train/beta1_power/readC_train/beta2_power/readC_train/Adam/learning_rateC_train/Adam/beta1C_train/Adam/beta2C_train/Adam/epsilonQC_train/gradients/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*/
_class%
#!loc:@Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
�
C_train/Adam/mulMulC_train/beta1_power/readC_train/Adam/beta14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
C_train/Adam/AssignAssignC_train/beta1_powerC_train/Adam/mul*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
C_train/Adam/mul_1MulC_train/beta2_power/readC_train/Adam/beta24^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam*
T0*(
_class
loc:@Critic/eval_net/l1/b1*
_output_shapes
: 
�
C_train/Adam/Assign_1AssignC_train/beta2_powerC_train/Adam/mul_1*
use_locking( *
T0*(
_class
loc:@Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
�
C_train/AdamNoOp^C_train/Adam/Assign^C_train/Adam/Assign_14^C_train/Adam/update_Critic/eval_net/l1/b1/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_a/ApplyAdam6^C_train/Adam/update_Critic/eval_net/l1/w1_s/ApplyAdam;^C_train/Adam/update_Critic/eval_net/q/dense/bias/ApplyAdam=^C_train/Adam/update_Critic/eval_net/q/dense/kernel/ApplyAdam
u
a_grad/gradients/ShapeShapeCritic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
_
a_grad/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
a_grad/gradients/FillFilla_grad/gradients/Shapea_grad/gradients/grad_ys_0*'
_output_shapes
:���������*
T0*

index_type0
�
Aa_grad/gradients/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrada_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
�
;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMula_grad/gradients/Fill#Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
=a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMulCritic/eval_net/l1/Relua_grad/gradients/Fill*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
�
6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad;a_grad/gradients/Critic/eval_net/q/dense/MatMul_grad/MatMulCritic/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/ShapeShapeCritic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
�
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
�
Da_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape6a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2a_grad/gradients/Critic/eval_net/l1/add_1_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradDa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeReshape2a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/Relu_grad/ReluGradFa_grad/gradients/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
8a_grad/gradients/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape4a_grad/gradients/Critic/eval_net/l1/add_1_grad/Sum_16a_grad/gradients/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
2a_grad/gradients/Critic/eval_net/l1/add_grad/ShapeShapeCritic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
�
4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1ShapeCritic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
Ba_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape4a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0a_grad/gradients/Critic/eval_net/l1/add_grad/SumSum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeBa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
4a_grad/gradients/Critic/eval_net/l1/add_grad/ReshapeReshape0a_grad/gradients/Critic/eval_net/l1/add_grad/Sum2a_grad/gradients/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_1Sum6a_grad/gradients/Critic/eval_net/l1/add_1_grad/ReshapeDa_grad/gradients/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Reshape2a_grad/gradients/Critic/eval_net/l1/add_grad/Sum_14a_grad/gradients/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
8a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1Critic/eval_net/l1/w1_a/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������
�
:a_grad/gradients/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMulCritic/StopGradient6a_grad/gradients/Critic/eval_net/l1/add_grad/Reshape_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
L
mul_8/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
^
mul_8Mulmul_8/xCritic/target_net/l1/w1_s/read*
_output_shapes

:P*
T0
L
mul_9/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
\
mul_9Mulmul_9/xCritic/eval_net/l1/w1_s/read*
_output_shapes

:P*
T0
C
add_4Addmul_8mul_9*
_output_shapes

:P*
T0
�
Assign_4AssignCritic/target_net/l1/w1_sadd_4*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:P
M
mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *�p}?
`
mul_10Mulmul_10/xCritic/target_net/l1/w1_a/read*
_output_shapes

:*
T0
M
mul_11/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
^
mul_11Mulmul_11/xCritic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
E
add_5Addmul_10mul_11*
T0*
_output_shapes

:
�
Assign_5AssignCritic/target_net/l1/w1_aadd_5*
use_locking(*
T0*,
_class"
 loc:@Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
M
mul_12/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
^
mul_12Mulmul_12/xCritic/target_net/l1/b1/read*
T0*
_output_shapes

:
M
mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
\
mul_13Mulmul_13/xCritic/eval_net/l1/b1/read*
T0*
_output_shapes

:
E
add_6Addmul_12mul_13*
T0*
_output_shapes

:
�
Assign_6AssignCritic/target_net/l1/b1add_6*
use_locking(*
T0**
_class 
loc:@Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
M
mul_14/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
g
mul_14Mulmul_14/x%Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
M
mul_15/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
e
mul_15Mulmul_15/x#Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
E
add_7Addmul_14mul_15*
_output_shapes

:*
T0
�
Assign_7Assign Critic/target_net/q/dense/kerneladd_7*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@Critic/target_net/q/dense/kernel*
validate_shape(
M
mul_16/xConst*
dtype0*
_output_shapes
: *
valueB
 *�p}?
a
mul_16Mulmul_16/x#Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
M
mul_17/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
_
mul_17Mulmul_17/x!Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
A
add_8Addmul_16mul_17*
_output_shapes
:*
T0
�
Assign_8AssignCritic/target_net/q/dense/biasadd_8*
use_locking(*
T0*1
_class'
%#loc:@Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
u
policy_grads/gradients/ShapeShapeActor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
e
 policy_grads/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
policy_grads/gradients/FillFillpolicy_grads/gradients/Shape policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:���������
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ShapeShapeActor/eval_net/a/a/Tanh*
T0*
out_type0*
_output_shapes
:
�
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Kpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulMulpolicy_grads/gradients/FillActor/eval_net/a/scaled_a/y*'
_output_shapes
:���������*
T0
�
9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/SumSum9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/MulKpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/ReshapeReshape9policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1MulActor/eval_net/a/a/Tanhpolicy_grads/gradients/Fill*
T0*'
_output_shapes
:���������
�
;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1Sum;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Mul_1Mpolicy_grads/gradients/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
?policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape;policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Sum_1=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradTanhGradActor/eval_net/a/a/Tanh=policy_grads/gradients/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:���������
�
Bpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGradActor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMulActor/eval_net/l1/Relu<policy_grads/gradients/Actor/eval_net/a/a/Tanh_grad/TanhGrad*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
�
;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradReluGrad<policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMulActor/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
Apolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
;policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMulMatMul;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGradActor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:���������P*
transpose_b(*
T0
�
=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s;policy_grads/gradients/Actor/eval_net/l1/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes

:P*
transpose_b( 
�
!A_train/beta1_power/initial_valueConst*
_output_shapes
: **
_class 
loc:@Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0
�
A_train/beta1_power
VariableV2**
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
A_train/beta1_power/AssignAssignA_train/beta1_power!A_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
�
A_train/beta1_power/readIdentityA_train/beta1_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
!A_train/beta2_power/initial_valueConst**
_class 
loc:@Actor/eval_net/a/a/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
A_train/beta2_power
VariableV2*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
A_train/beta2_power/AssignAssignA_train/beta2_power!A_train/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
�
A_train/beta2_power/readIdentityA_train/beta2_power*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
GA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
�
=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0
�
7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillGA_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensor=A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@Actor/eval_net/l1/kernel*
_output_shapes

:P
�
%A_train/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0*
_output_shapes

:P
�
,A_train/Actor/eval_net/l1/kernel/Adam/AssignAssign%A_train/Actor/eval_net/l1/kernel/Adam7A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P*
use_locking(*
T0
�
*A_train/Actor/eval_net/l1/kernel/Adam/readIdentity%A_train/Actor/eval_net/l1/kernel/Adam*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
IA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"P      *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0
�
?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
�
9A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillIA_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensor?A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:P*
T0*

index_type0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
'A_train/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
_output_shapes

:P*
shared_name *+
_class!
loc:@Actor/eval_net/l1/kernel*
	container *
shape
:P*
dtype0
�
.A_train/Actor/eval_net/l1/kernel/Adam_1/AssignAssign'A_train/Actor/eval_net/l1/kernel/Adam_19A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:P
�
,A_train/Actor/eval_net/l1/kernel/Adam_1/readIdentity'A_train/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes

:P*
T0*+
_class!
loc:@Actor/eval_net/l1/kernel
�
5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
valueB*    *)
_class
loc:@Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
�
#A_train/Actor/eval_net/l1/bias/Adam
VariableV2*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
*A_train/Actor/eval_net/l1/bias/Adam/AssignAssign#A_train/Actor/eval_net/l1/bias/Adam5A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
(A_train/Actor/eval_net/l1/bias/Adam/readIdentity#A_train/Actor/eval_net/l1/bias/Adam*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
_output_shapes
:
�
7A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueB*    *)
_class
loc:@Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
�
%A_train/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Actor/eval_net/l1/bias*
	container *
shape:
�
,A_train/Actor/eval_net/l1/bias/Adam_1/AssignAssign%A_train/Actor/eval_net/l1/bias/Adam_17A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
�
*A_train/Actor/eval_net/l1/bias/Adam_1/readIdentity%A_train/Actor/eval_net/l1/bias/Adam_1*
_output_shapes
:*
T0*)
_class
loc:@Actor/eval_net/l1/bias
�
8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
�
&A_train/Actor/eval_net/a/a/kernel/Adam
VariableV2*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
�
-A_train/Actor/eval_net/a/a/kernel/Adam/AssignAssign&A_train/Actor/eval_net/a/a/kernel/Adam8A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
+A_train/Actor/eval_net/a/a/kernel/Adam/readIdentity&A_train/Actor/eval_net/a/a/kernel/Adam*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
_output_shapes

:
�
:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
�
(A_train/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@Actor/eval_net/a/a/kernel*
	container *
shape
:
�
/A_train/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign(A_train/Actor/eval_net/a/a/kernel/Adam_1:A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
-A_train/Actor/eval_net/a/a/kernel/Adam_1/readIdentity(A_train/Actor/eval_net/a/a/kernel/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel
�
6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
�
$A_train/Actor/eval_net/a/a/bias/Adam
VariableV2*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
+A_train/Actor/eval_net/a/a/bias/Adam/AssignAssign$A_train/Actor/eval_net/a/a/bias/Adam6A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
�
)A_train/Actor/eval_net/a/a/bias/Adam/readIdentity$A_train/Actor/eval_net/a/a/bias/Adam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
:
�
8A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    **
_class 
loc:@Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
�
&A_train/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Actor/eval_net/a/a/bias*
	container *
shape:
�
-A_train/Actor/eval_net/a/a/bias/Adam_1/AssignAssign&A_train/Actor/eval_net/a/a/bias/Adam_18A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
�
+A_train/Actor/eval_net/a/a/bias/Adam_1/readIdentity&A_train/Actor/eval_net/a/a/bias/Adam_1*
_output_shapes
:*
T0**
_class 
loc:@Actor/eval_net/a/a/bias
_
A_train/Adam/learning_rateConst*
valueB
 *o��*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
W
A_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
Y
A_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
6A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdamActor/eval_net/l1/kernel%A_train/Actor/eval_net/l1/kernel/Adam'A_train/Actor/eval_net/l1/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon=policy_grads/gradients/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:P
�
4A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam	ApplyAdamActor/eval_net/l1/bias#A_train/Actor/eval_net/l1/bias/Adam%A_train/Actor/eval_net/l1/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonApolicy_grads/gradients/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
use_locking( *
T0*)
_class
loc:@Actor/eval_net/l1/bias*
use_nesterov( 
�
7A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdamActor/eval_net/a/a/kernel&A_train/Actor/eval_net/a/a/kernel/Adam(A_train/Actor/eval_net/a/a/kernel/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilon>policy_grads/gradients/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*,
_class"
 loc:@Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:
�
5A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdamActor/eval_net/a/a/bias$A_train/Actor/eval_net/a/a/bias/Adam&A_train/Actor/eval_net/a/a/bias/Adam_1A_train/beta1_power/readA_train/beta2_power/readA_train/Adam/learning_rateA_train/Adam/beta1A_train/Adam/beta2A_train/Adam/epsilonBpolicy_grads/gradients/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
�
A_train/Adam/mulMulA_train/beta1_power/readA_train/Adam/beta16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
�
A_train/Adam/AssignAssignA_train/beta1_powerA_train/Adam/mul**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
A_train/Adam/mul_1MulA_train/beta2_power/readA_train/Adam/beta26^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam*
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
_output_shapes
: 
�
A_train/Adam/Assign_1AssignA_train/beta2_powerA_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
�
A_train/AdamNoOp^A_train/Adam/Assign^A_train/Adam/Assign_16^A_train/Adam/update_Actor/eval_net/a/a/bias/ApplyAdam8^A_train/Adam/update_Actor/eval_net/a/a/kernel/ApplyAdam5^A_train/Adam/update_Actor/eval_net/l1/bias/ApplyAdam7^A_train/Adam/update_Actor/eval_net/l1/kernel/ApplyAdam
�
initNoOp,^A_train/Actor/eval_net/a/a/bias/Adam/Assign.^A_train/Actor/eval_net/a/a/bias/Adam_1/Assign.^A_train/Actor/eval_net/a/a/kernel/Adam/Assign0^A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign+^A_train/Actor/eval_net/l1/bias/Adam/Assign-^A_train/Actor/eval_net/l1/bias/Adam_1/Assign-^A_train/Actor/eval_net/l1/kernel/Adam/Assign/^A_train/Actor/eval_net/l1/kernel/Adam_1/Assign^A_train/beta1_power/Assign^A_train/beta2_power/Assign^Actor/eval_net/a/a/bias/Assign!^Actor/eval_net/a/a/kernel/Assign^Actor/eval_net/l1/bias/Assign ^Actor/eval_net/l1/kernel/Assign!^Actor/target_net/a/a/bias/Assign#^Actor/target_net/a/a/kernel/Assign ^Actor/target_net/l1/bias/Assign"^Actor/target_net/l1/kernel/Assign*^C_train/Critic/eval_net/l1/b1/Adam/Assign,^C_train/Critic/eval_net/l1/b1/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_a/Adam/Assign.^C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign,^C_train/Critic/eval_net/l1/w1_s/Adam/Assign.^C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign1^C_train/Critic/eval_net/q/dense/bias/Adam/Assign3^C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign3^C_train/Critic/eval_net/q/dense/kernel/Adam/Assign5^C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign^C_train/beta1_power/Assign^C_train/beta2_power/Assign^Critic/eval_net/l1/b1/Assign^Critic/eval_net/l1/w1_a/Assign^Critic/eval_net/l1/w1_s/Assign$^Critic/eval_net/q/dense/bias/Assign&^Critic/eval_net/q/dense/kernel/Assign^Critic/target_net/l1/b1/Assign!^Critic/target_net/l1/w1_a/Assign!^Critic/target_net/l1/w1_s/Assign&^Critic/target_net/q/dense/bias/Assign(^Critic/target_net/q/dense/kernel/Assign"&"�

trainable_variables�
�

�
Actor/eval_net/l1/kernel:0Actor/eval_net/l1/kernel/AssignActor/eval_net/l1/kernel/read:024Actor/eval_net/l1/kernel/Initializer/random_normal:08
�
Actor/eval_net/l1/bias:0Actor/eval_net/l1/bias/AssignActor/eval_net/l1/bias/read:02*Actor/eval_net/l1/bias/Initializer/Const:08
�
Actor/eval_net/a/a/kernel:0 Actor/eval_net/a/a/kernel/Assign Actor/eval_net/a/a/kernel/read:025Actor/eval_net/a/a/kernel/Initializer/random_normal:08
�
Actor/eval_net/a/a/bias:0Actor/eval_net/a/a/bias/AssignActor/eval_net/a/a/bias/read:02+Actor/eval_net/a/a/bias/Initializer/Const:08
�
Critic/eval_net/l1/w1_s:0Critic/eval_net/l1/w1_s/AssignCritic/eval_net/l1/w1_s/read:023Critic/eval_net/l1/w1_s/Initializer/random_normal:08
�
Critic/eval_net/l1/w1_a:0Critic/eval_net/l1/w1_a/AssignCritic/eval_net/l1/w1_a/read:023Critic/eval_net/l1/w1_a/Initializer/random_normal:08
�
Critic/eval_net/l1/b1:0Critic/eval_net/l1/b1/AssignCritic/eval_net/l1/b1/read:02)Critic/eval_net/l1/b1/Initializer/Const:08
�
 Critic/eval_net/q/dense/kernel:0%Critic/eval_net/q/dense/kernel/Assign%Critic/eval_net/q/dense/kernel/read:02:Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
�
Critic/eval_net/q/dense/bias:0#Critic/eval_net/q/dense/bias/Assign#Critic/eval_net/q/dense/bias/read:020Critic/eval_net/q/dense/bias/Initializer/Const:08"*
train_op

C_train/Adam
A_train/Adam"�5
	variables�5�5
�
Actor/eval_net/l1/kernel:0Actor/eval_net/l1/kernel/AssignActor/eval_net/l1/kernel/read:024Actor/eval_net/l1/kernel/Initializer/random_normal:08
�
Actor/eval_net/l1/bias:0Actor/eval_net/l1/bias/AssignActor/eval_net/l1/bias/read:02*Actor/eval_net/l1/bias/Initializer/Const:08
�
Actor/eval_net/a/a/kernel:0 Actor/eval_net/a/a/kernel/Assign Actor/eval_net/a/a/kernel/read:025Actor/eval_net/a/a/kernel/Initializer/random_normal:08
�
Actor/eval_net/a/a/bias:0Actor/eval_net/a/a/bias/AssignActor/eval_net/a/a/bias/read:02+Actor/eval_net/a/a/bias/Initializer/Const:08
�
Actor/target_net/l1/kernel:0!Actor/target_net/l1/kernel/Assign!Actor/target_net/l1/kernel/read:026Actor/target_net/l1/kernel/Initializer/random_normal:0
�
Actor/target_net/l1/bias:0Actor/target_net/l1/bias/AssignActor/target_net/l1/bias/read:02,Actor/target_net/l1/bias/Initializer/Const:0
�
Actor/target_net/a/a/kernel:0"Actor/target_net/a/a/kernel/Assign"Actor/target_net/a/a/kernel/read:027Actor/target_net/a/a/kernel/Initializer/random_normal:0
�
Actor/target_net/a/a/bias:0 Actor/target_net/a/a/bias/Assign Actor/target_net/a/a/bias/read:02-Actor/target_net/a/a/bias/Initializer/Const:0
�
Critic/eval_net/l1/w1_s:0Critic/eval_net/l1/w1_s/AssignCritic/eval_net/l1/w1_s/read:023Critic/eval_net/l1/w1_s/Initializer/random_normal:08
�
Critic/eval_net/l1/w1_a:0Critic/eval_net/l1/w1_a/AssignCritic/eval_net/l1/w1_a/read:023Critic/eval_net/l1/w1_a/Initializer/random_normal:08
�
Critic/eval_net/l1/b1:0Critic/eval_net/l1/b1/AssignCritic/eval_net/l1/b1/read:02)Critic/eval_net/l1/b1/Initializer/Const:08
�
 Critic/eval_net/q/dense/kernel:0%Critic/eval_net/q/dense/kernel/Assign%Critic/eval_net/q/dense/kernel/read:02:Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
�
Critic/eval_net/q/dense/bias:0#Critic/eval_net/q/dense/bias/Assign#Critic/eval_net/q/dense/bias/read:020Critic/eval_net/q/dense/bias/Initializer/Const:08
�
Critic/target_net/l1/w1_s:0 Critic/target_net/l1/w1_s/Assign Critic/target_net/l1/w1_s/read:025Critic/target_net/l1/w1_s/Initializer/random_normal:0
�
Critic/target_net/l1/w1_a:0 Critic/target_net/l1/w1_a/Assign Critic/target_net/l1/w1_a/read:025Critic/target_net/l1/w1_a/Initializer/random_normal:0
�
Critic/target_net/l1/b1:0Critic/target_net/l1/b1/AssignCritic/target_net/l1/b1/read:02+Critic/target_net/l1/b1/Initializer/Const:0
�
"Critic/target_net/q/dense/kernel:0'Critic/target_net/q/dense/kernel/Assign'Critic/target_net/q/dense/kernel/read:02<Critic/target_net/q/dense/kernel/Initializer/random_normal:0
�
 Critic/target_net/q/dense/bias:0%Critic/target_net/q/dense/bias/Assign%Critic/target_net/q/dense/bias/read:022Critic/target_net/q/dense/bias/Initializer/Const:0
t
C_train/beta1_power:0C_train/beta1_power/AssignC_train/beta1_power/read:02#C_train/beta1_power/initial_value:0
t
C_train/beta2_power:0C_train/beta2_power/AssignC_train/beta2_power/read:02#C_train/beta2_power/initial_value:0
�
&C_train/Critic/eval_net/l1/w1_s/Adam:0+C_train/Critic/eval_net/l1/w1_s/Adam/Assign+C_train/Critic/eval_net/l1/w1_s/Adam/read:028C_train/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
�
(C_train/Critic/eval_net/l1/w1_s/Adam_1:0-C_train/Critic/eval_net/l1/w1_s/Adam_1/Assign-C_train/Critic/eval_net/l1/w1_s/Adam_1/read:02:C_train/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
�
&C_train/Critic/eval_net/l1/w1_a/Adam:0+C_train/Critic/eval_net/l1/w1_a/Adam/Assign+C_train/Critic/eval_net/l1/w1_a/Adam/read:028C_train/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
�
(C_train/Critic/eval_net/l1/w1_a/Adam_1:0-C_train/Critic/eval_net/l1/w1_a/Adam_1/Assign-C_train/Critic/eval_net/l1/w1_a/Adam_1/read:02:C_train/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
�
$C_train/Critic/eval_net/l1/b1/Adam:0)C_train/Critic/eval_net/l1/b1/Adam/Assign)C_train/Critic/eval_net/l1/b1/Adam/read:026C_train/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
�
&C_train/Critic/eval_net/l1/b1/Adam_1:0+C_train/Critic/eval_net/l1/b1/Adam_1/Assign+C_train/Critic/eval_net/l1/b1/Adam_1/read:028C_train/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
�
-C_train/Critic/eval_net/q/dense/kernel/Adam:02C_train/Critic/eval_net/q/dense/kernel/Adam/Assign2C_train/Critic/eval_net/q/dense/kernel/Adam/read:02?C_train/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
�
/C_train/Critic/eval_net/q/dense/kernel/Adam_1:04C_train/Critic/eval_net/q/dense/kernel/Adam_1/Assign4C_train/Critic/eval_net/q/dense/kernel/Adam_1/read:02AC_train/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
�
+C_train/Critic/eval_net/q/dense/bias/Adam:00C_train/Critic/eval_net/q/dense/bias/Adam/Assign0C_train/Critic/eval_net/q/dense/bias/Adam/read:02=C_train/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
�
-C_train/Critic/eval_net/q/dense/bias/Adam_1:02C_train/Critic/eval_net/q/dense/bias/Adam_1/Assign2C_train/Critic/eval_net/q/dense/bias/Adam_1/read:02?C_train/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
t
A_train/beta1_power:0A_train/beta1_power/AssignA_train/beta1_power/read:02#A_train/beta1_power/initial_value:0
t
A_train/beta2_power:0A_train/beta2_power/AssignA_train/beta2_power/read:02#A_train/beta2_power/initial_value:0
�
'A_train/Actor/eval_net/l1/kernel/Adam:0,A_train/Actor/eval_net/l1/kernel/Adam/Assign,A_train/Actor/eval_net/l1/kernel/Adam/read:029A_train/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
�
)A_train/Actor/eval_net/l1/kernel/Adam_1:0.A_train/Actor/eval_net/l1/kernel/Adam_1/Assign.A_train/Actor/eval_net/l1/kernel/Adam_1/read:02;A_train/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
�
%A_train/Actor/eval_net/l1/bias/Adam:0*A_train/Actor/eval_net/l1/bias/Adam/Assign*A_train/Actor/eval_net/l1/bias/Adam/read:027A_train/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
�
'A_train/Actor/eval_net/l1/bias/Adam_1:0,A_train/Actor/eval_net/l1/bias/Adam_1/Assign,A_train/Actor/eval_net/l1/bias/Adam_1/read:029A_train/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
�
(A_train/Actor/eval_net/a/a/kernel/Adam:0-A_train/Actor/eval_net/a/a/kernel/Adam/Assign-A_train/Actor/eval_net/a/a/kernel/Adam/read:02:A_train/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
�
*A_train/Actor/eval_net/a/a/kernel/Adam_1:0/A_train/Actor/eval_net/a/a/kernel/Adam_1/Assign/A_train/Actor/eval_net/a/a/kernel/Adam_1/read:02<A_train/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
�
&A_train/Actor/eval_net/a/a/bias/Adam:0+A_train/Actor/eval_net/a/a/bias/Adam/Assign+A_train/Actor/eval_net/a/a/bias/Adam/read:028A_train/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
�
(A_train/Actor/eval_net/a/a/bias/Adam_1:0-A_train/Actor/eval_net/a/a/bias/Adam_1/Assign-A_train/Actor/eval_net/a/a/bias/Adam_1/read:02:A_train/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0ZtV�