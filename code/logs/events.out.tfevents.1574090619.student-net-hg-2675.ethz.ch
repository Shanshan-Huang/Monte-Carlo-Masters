       ЃK"	  Ро­tзAbrain.Event:2ЅЧ0Цп     [От	Тlто­tзA"О
f
S/sPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
f
R/rPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
S_/s_Placeholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
К
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"      *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
­
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Џ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0

I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2

80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:

40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:*
T0
Н
0/Actor/eval_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:
ў
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(

0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:
Є
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
Б
0/Actor/eval_net/l1/bias
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ъ
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:

0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:*
T0
Ђ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Њ
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
М
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Џ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0
Б
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel

J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
seed2
Ѓ
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel

50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:
П
0/Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:

"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ђ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:
І
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*ЭЬЬ=*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0
Г
0/Actor/eval_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:
ю
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:

0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
Й
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
­
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
w
0/Actor/eval_net/a/a/SigmoidSigmoid0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
b
0/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Sigmoid0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ
О
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"      */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
Б
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0
Г
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
Ё
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(
Ї
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel

60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:
С
0/Actor/target_net/l1/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel

#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:
Ѕ
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
_output_shapes

:*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
Ј
,0/Actor/target_net/l1/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*-
_class#
!loc:@0/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:
Е
0/Actor/target_net/l1/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@0/Actor/target_net/l1/bias*
	container 
ђ
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:

0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:
Ј
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"      *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
:
Г
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Е
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Є
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed28
Ћ
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:

70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:
У
0/Actor/target_net/a/a/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
	container 

$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ј
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
_output_shapes

:*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
Њ
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
З
0/Actor/target_net/a/a/bias
VariableV2*
shared_name *.
_class$
" loc:@0/Actor/target_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
і
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:

 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
П
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Г
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
{
0/Actor/target_net/a/a/SigmoidSigmoid0/Actor/target_net/a/a/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Sigmoid0/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ
L
0/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
a
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
T0*
_output_shapes

:
N
	0/mul_1/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
c
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:
E
0/addAdd0/mul0/mul_1*
T0*
_output_shapes

:
К
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:
N
	0/mul_2/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:
N
	0/mul_3/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
]
0/mul_3Mul	0/mul_3/x0/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:
E
0/add_1Add0/mul_20/mul_3*
_output_shapes
:*
T0
Ж

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
N
	0/mul_4/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
0/mul_4Mul	0/mul_4/x"0/Actor/target_net/a/a/kernel/read*
_output_shapes

:*
T0
N
	0/mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
d
0/mul_5Mul	0/mul_5/x 0/Actor/eval_net/a/a/kernel/read*
_output_shapes

:*
T0
I
0/add_2Add0/mul_40/mul_5*
T0*
_output_shapes

:
Р

0/Assign_2Assign0/Actor/target_net/a/a/kernel0/add_2*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
N
	0/mul_6/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
`
0/mul_6Mul	0/mul_6/x 0/Actor/target_net/a/a/bias/read*
_output_shapes
:*
T0
N
	0/mul_7/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
^
0/mul_7Mul	0/mul_7/x0/Actor/eval_net/a/a/bias/read*
_output_shapes
:*
T0
E
0/add_3Add0/mul_60/mul_7*
T0*
_output_shapes
:
И

0/Assign_3Assign0/Actor/target_net/a/a/bias0/add_3*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(
t
0/Critic/StopGradientStopGradient0/Actor/eval_net/a/scaled_a*
T0*'
_output_shapes
:џџџџџџџџџ
И
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0
Ћ
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
­
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s

H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2c*
dtype0

70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:

30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
Л
0/Critic/eval_net/l1/w1_s
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
њ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:

0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
И
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ћ
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
­
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 

H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:*

seed

70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:

30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Л
0/Critic/eval_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container *
shape
:
њ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(

0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Њ
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*ЭЬЬ=**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
З
0/Critic/eval_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape
:
ъ
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:

0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ђ
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ж
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ

0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*'
_output_shapes
:џџџџџџџџџ*
T0
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
Й
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Л
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
­
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
seed2~*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
З
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
 
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
Щ
 0/Critic/eval_net/q/dense/kernel
VariableV2*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0

'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
Б
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
А
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Н
0/Critic/eval_net/q/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias

%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Ї
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ф
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
М
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *.
_class$
" loc:@0/Critic/target_net/l1/w1_s
Џ
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
Б
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 

J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2
Ѓ
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0

50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
П
0/Critic/target_net/l1/w1_s
VariableV2*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:

"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
Ђ
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
М
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
Џ
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Б
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 

J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2
Ѓ
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:

50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
П
0/Critic/target_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
	container *
shape
:

"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(
Ђ
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
Ў
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*ЭЬЬ=*,
_class"
 loc:@0/Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
Л
0/Critic/target_net/l1/b1
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/target_net/l1/b1*
	container *
shape
:*
dtype0
ђ
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0

0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
_output_shapes

:
Ј
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Т
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ

0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ъ
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
Н
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
П
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Д
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2Ј
П
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ј
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Э
"0/Critic/target_net/q/dense/kernel
VariableV2*
_output_shapes

:*
shared_name *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0

)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
З
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
Д
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
С
 0/Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
	container *
shape:

'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias
­
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0
Ъ
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Т
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
U
0/target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
~
0/target_q/mulMul0/target_q/mul/x#0/Critic/target_net/q/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
\
0/target_q/addAddR/r0/target_q/mul*'
_output_shapes
:џџџџџџџџџ*
T0

0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
0/C_train/gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
b
0/C_train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 

60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Ф
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
л
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

00/C_train/gradients/0/TD_error/Mean_grad/Shape_1Shape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
s
00/C_train/gradients/0/TD_error/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
x
.0/C_train/gradients/0/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
-0/C_train/gradients/0/TD_error/Mean_grad/ProdProd00/C_train/gradients/0/TD_error/Mean_grad/Shape_1.0/C_train/gradients/0/TD_error/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
й
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
t
20/C_train/gradients/0/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
С
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
П
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ј
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
Ы
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
_output_shapes
:*
T0*
out_type0

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:

K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Д
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
т
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
и
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Г
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
ц
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
р
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg
у
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
я
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
џ
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
ї
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
к
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
є
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
ё
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
ш
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0

;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
Ъ
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
о
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
л
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:*
T0

70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:

90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:

G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
є
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
ж
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
м
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ч
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
Ы
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
р
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
н
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:

=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
б
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ш
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul
х
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:

#0/C_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ѕ
0/C_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: 
и
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 

0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 

#0/C_train/beta2_power/initial_valueConst*
valueB
 *wО?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ѕ
0/C_train/beta2_power
VariableV2**
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
и
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(

0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Н
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0*
_output_shapes

:
Ъ
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:

/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
К
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
П
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
_output_shapes

:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0
Ь
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:
Ѕ
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(
О
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
Н
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
Ъ
(0/C_train/0/Critic/eval_net/l1/w1_a/Adam
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0

/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
К
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
П
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
Ь
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ѕ
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
О
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
Й
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    
Ц
&0/C_train/0/Critic/eval_net/l1/b1/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1

-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Д
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Л
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    
Ш
(0/C_train/0/Critic/eval_net/l1/b1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape
:

/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
И
-0/C_train/0/Critic/eval_net/l1/b1/Adam_1/readIdentity(0/C_train/0/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ы
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
и
/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container *
shape
:
Л
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Я
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
Э
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
к
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container 
С
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
г
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
П
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0
Ь
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Џ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
Х
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
С
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    
Ю
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
Е
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Щ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
0/C_train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o:
Y
0/C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
0/C_train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
[
0/C_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
 
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
Ђ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a

70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
Ш
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
Л
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
Ю
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
Р
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
а
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
Ф
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
ў
0/C_train/AdamNoOp^0/C_train/Adam/Assign^0/C_train/Adam/Assign_18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam
y
0/a_grad/gradients/ShapeShape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
a
0/a_grad/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
Љ
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
с
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ю
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
д
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:

:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      

H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:

80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:

F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
џ
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ё
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ъ
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
N
	0/mul_8/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
d
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	0/mul_9/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes

:
I
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes

:
М

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
O

0/mul_10/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
0/mul_10Mul
0/mul_10/x 0/Critic/target_net/l1/w1_a/read*
T0*
_output_shapes

:
O

0/mul_11/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
0/mul_11Mul
0/mul_11/x0/Critic/eval_net/l1/w1_a/read*
T0*
_output_shapes

:
K
0/add_5Add0/mul_100/mul_11*
T0*
_output_shapes

:
М

0/Assign_5Assign0/Critic/target_net/l1/w1_a0/add_5*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
O

0/mul_12/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
d
0/mul_12Mul
0/mul_12/x0/Critic/target_net/l1/b1/read*
T0*
_output_shapes

:
O

0/mul_13/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
0/mul_13Mul
0/mul_13/x0/Critic/eval_net/l1/b1/read*
_output_shapes

:*
T0
K
0/add_6Add0/mul_120/mul_13*
T0*
_output_shapes

:
И

0/Assign_6Assign0/Critic/target_net/l1/b10/add_6*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
O

0/mul_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

0/mul_15/xConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
k
0/mul_15Mul
0/mul_15/x%0/Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
K
0/add_7Add0/mul_140/mul_15*
_output_shapes

:*
T0
Ъ

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(
O

0/mul_16/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
g
0/mul_16Mul
0/mul_16/x%0/Critic/target_net/q/dense/bias/read*
_output_shapes
:*
T0
O

0/mul_17/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
e
0/mul_17Mul
0/mul_17/x#0/Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
G
0/add_8Add0/mul_160/mul_17*
_output_shapes
:*
T0
Т

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
g
"0/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
­
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Sigmoid*
T0*
out_type0*
_output_shapes
:

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Љ
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Д
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ

=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Е
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Sigmoid0/policy_grads/gradients/Fill*
T0*'
_output_shapes
:џџџџџџџџџ

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ш
F0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad0/Actor/eval_net/a/a/SigmoidA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
й
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0

@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 0/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
§
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/TanhF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
й
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
б
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:

?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
р
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0

#0/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Ї
0/A_train/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container 
к
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 

#0/A_train/beta2_power/initial_valueConst*
valueB
 *wО?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Ї
0/A_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: 
к
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
П
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB*    *
dtype0
Ь
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container 
Ѓ
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Н
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:
С
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB*    
Ю
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Љ
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:
С
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes

:*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Г
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
Р
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(
Г
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:
Е
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0

00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
З
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:
С
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB*    
Ю
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Ї
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Р
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
_output_shapes

:*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
У
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
а
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container 
­
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ф
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:*
T0
Е
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
(0/A_train/0/Actor/eval_net/a/a/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0

/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ж
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
З
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    
Ф
*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ё
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
К
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
a
0/A_train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *oК
Y
0/A_train/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Y
0/A_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
0/A_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 

80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias

;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel

90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( 

0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
Т
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
К
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
К
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"      *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
­
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Џ
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 

I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2Т*
dtype0*
_output_shapes

:*

seed*
T0

81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:*
T0

41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:
Н
1/Actor/eval_net/l1/kernel
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
ў
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(

1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*
_output_shapes

:*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Є
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
Б
1/Actor/eval_net/l1/bias
VariableV2*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ъ
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:
Ђ
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Њ
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
М
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
Џ
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
Б
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 

J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
seed2в*
dtype0*
_output_shapes

:*

seed
Ѓ
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel

51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:
П
1/Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:

"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ђ
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*
_output_shapes

:*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
І
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Г
1/Actor/eval_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:
ю
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:

1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
Й
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
­
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
b
1/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
О
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"      */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
Б
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *    */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
Г
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
Ђ
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
seed2ф
Ї
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:

61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:
С
1/Actor/target_net/l1/kernel
VariableV2*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:

#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:
Ѕ
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
_output_shapes

:*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
Ј
,1/Actor/target_net/l1/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*-
_class#
!loc:@1/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:
Е
1/Actor/target_net/l1/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@1/Actor/target_net/l1/bias*
	container 
ђ
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:

1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
_output_shapes
:
Ј
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
Г
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
Е
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Ѕ
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed2є*
dtype0*
_output_shapes

:*

seed
Ћ
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:

71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:
У
1/Actor/target_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container *
shape
:

$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ј
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:
Њ
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
З
1/Actor/target_net/a/a/bias
VariableV2*
_output_shapes
:*
shared_name *.
_class$
" loc:@1/Actor/target_net/a/a/bias*
	container *
shape:*
dtype0
і
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:

 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*
_output_shapes
:*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias
П
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Г
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
d
1/Actor/target_net/a/scaled_a/yConst*
_output_shapes
: *
valueB
 *  HC*
dtype0

1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
L
1/mul/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
a
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
T0*
_output_shapes

:
N
	1/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
c
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:
E
1/addAdd1/mul1/mul_1*
_output_shapes

:*
T0
К
1/AssignAssign1/Actor/target_net/l1/kernel1/add*
_output_shapes

:*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(
N
	1/mul_2/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
_output_shapes
:*
T0
N
	1/mul_3/xConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
]
1/mul_3Mul	1/mul_3/x1/Actor/eval_net/l1/bias/read*
_output_shapes
:*
T0
E
1/add_1Add1/mul_21/mul_3*
_output_shapes
:*
T0
Ж

1/Assign_1Assign1/Actor/target_net/l1/bias1/add_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias
N
	1/mul_4/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
_output_shapes

:*
T0
N
	1/mul_5/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
1/mul_5Mul	1/mul_5/x 1/Actor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:
I
1/add_2Add1/mul_41/mul_5*
T0*
_output_shapes

:
Р

1/Assign_2Assign1/Actor/target_net/a/a/kernel1/add_2*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
N
	1/mul_6/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
`
1/mul_6Mul	1/mul_6/x 1/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	1/mul_7/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
^
1/mul_7Mul	1/mul_7/x1/Actor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
E
1/add_3Add1/mul_61/mul_7*
T0*
_output_shapes
:
И

1/Assign_3Assign1/Actor/target_net/a/a/bias1/add_3*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
t
1/Critic/StopGradientStopGradient1/Actor/eval_net/a/scaled_a*
T0*'
_output_shapes
:џџџџџџџџџ
И
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
Ћ
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
­
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 

H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s

71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:

31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
Л
1/Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:
њ
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(

1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
И
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ћ
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
­
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0

H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
seed2Ј*
dtype0

71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:

31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
Л
1/Critic/eval_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:
њ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(

1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
Њ
)1/Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*ЭЬЬ=**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
З
1/Critic/eval_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:
ъ
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0

1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:*
T0
Ђ
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ж
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 

1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ

1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*'
_output_shapes
:џџџџџџџџџ*
T0
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ц
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
Й
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Л
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Ў
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
seed2К
З
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
 
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Щ
 1/Critic/eval_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container *
shape
:

'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Б
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
А
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*ЭЬЬ=*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
Н
1/Critic/eval_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:

%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ї
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ф
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
М
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
Џ
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
Б
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 

J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
seed2Щ*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s
Ѓ
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:

51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
П
1/Critic/target_net/l1/w1_s
VariableV2*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 

"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
Ђ
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s
М
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0
Џ
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Б
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 

J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2в*
dtype0*
_output_shapes

:*

seed
Ѓ
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:

51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
П
1/Critic/target_net/l1/w1_a
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_a

"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ђ
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a
Ў
+1/Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*ЭЬЬ=*,
_class"
 loc:@1/Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
Л
1/Critic/target_net/l1/b1
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/target_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
ђ
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(

1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
_output_shapes

:
Ј
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Т
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
T0

1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
Н
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
П
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЭЬЬ=*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Д
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*

seed*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2ф*
dtype0*
_output_shapes

:
П
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel
Ј
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
Э
"1/Critic/target_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
	container *
shape
:

)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
З
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
Д
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*ЭЬЬ=*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
С
 1/Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
	container *
shape:

'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
­
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0
Ъ
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Т
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
U
1/target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
~
1/target_q/mulMul1/target_q/mul/x#1/Critic/target_net/q/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
\
1/target_q/addAddR/r1/target_q/mul*
T0*'
_output_shapes
:џџџџџџџџџ

1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
a
1/TD_error/ConstConst*
_output_shapes
:*
valueB"       *
dtype0

1/TD_error/MeanMean1/TD_error/SquaredDifference1/TD_error/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
\
1/C_train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
b
1/C_train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0

61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ф
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
л
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

01/C_train/gradients/1/TD_error/Mean_grad/Shape_1Shape1/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
s
01/C_train/gradients/1/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
x
.1/C_train/gradients/1/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
й
/1/C_train/gradients/1/TD_error/Mean_grad/Prod_1Prod01/C_train/gradients/1/TD_error/Mean_grad/Shape_201/C_train/gradients/1/TD_error/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
t
21/C_train/gradients/1/TD_error/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
С
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
П
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ј
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Ы
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
_output_shapes
:*
T0*
out_type0

=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0

K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
т
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
и
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Г
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
ц
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
р
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ*
T0
у
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
я
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
џ
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
ї
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
к
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
є
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
ё
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
ш
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:

;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Ъ
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
о
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape
л
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:

71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0

91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0

G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
ж
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
м
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1

;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ч
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Ы
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
р
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
н
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:

=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
§
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
б
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ш
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul
х
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:

#1/C_train/beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0
Ѕ
1/C_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: 
и
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(

1/C_train/beta1_power/readIdentity1/C_train/beta1_power*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 

#1/C_train/beta2_power/initial_valueConst*
valueB
 *wО?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ѕ
1/C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: 
и
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 

1/C_train/beta2_power/readIdentity1/C_train/beta2_power*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Н
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0*
_output_shapes

:
Ъ
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0

/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
К
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
П
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB*    
Ь
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ѕ
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
О
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
Н
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
Ъ
(1/C_train/1/Critic/eval_net/l1/w1_a/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a

/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
К
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
П
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    
Ь
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
Ѕ
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
О
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
Й
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    
Ц
&1/C_train/1/Critic/eval_net/l1/b1/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1

-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(
Д
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Л
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
Ш
(1/C_train/1/Critic/eval_net/l1/b1/Adam_1
VariableV2*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:

/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/AssignAssign(1/C_train/1/Critic/eval_net/l1/b1/Adam_1:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
И
-1/C_train/1/Critic/eval_net/l1/b1/Adam_1/readIdentity(1/C_train/1/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ы
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
и
/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container 
Л
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Я
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Э
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
к
11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container *
shape
:
С
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(
г
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
П
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ь
-1/C_train/1/Critic/eval_net/q/dense/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
Џ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Х
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
С
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ю
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
Е
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(
Щ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
a
1/C_train/Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Y
1/C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
1/C_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
 
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:
Ђ
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:

71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
Ш
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
Л
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
Ю
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Р
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
а
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
Ф
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
ў
1/C_train/AdamNoOp^1/C_train/Adam/Assign^1/C_train/Adam/Assign_18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam
y
1/a_grad/gradients/ShapeShape!1/Critic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
a
1/a_grad/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Љ
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
с
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ю
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
д
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*'
_output_shapes
:џџџџџџџџџ*
T0

81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:

:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0

61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0

81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:

F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
џ
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ё
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
њ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ъ
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
N
	1/mul_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
d
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	1/mul_9/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
_output_shapes

:*
T0
I
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes

:
М

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
O

1/mul_10/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
1/mul_10Mul
1/mul_10/x 1/Critic/target_net/l1/w1_a/read*
T0*
_output_shapes

:
O

1/mul_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
d
1/mul_11Mul
1/mul_11/x1/Critic/eval_net/l1/w1_a/read*
T0*
_output_shapes

:
K
1/add_5Add1/mul_101/mul_11*
T0*
_output_shapes

:
М

1/Assign_5Assign1/Critic/target_net/l1/w1_a1/add_5*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
O

1/mul_12/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
d
1/mul_12Mul
1/mul_12/x1/Critic/target_net/l1/b1/read*
T0*
_output_shapes

:
O

1/mul_13/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
1/mul_13Mul
1/mul_13/x1/Critic/eval_net/l1/b1/read*
T0*
_output_shapes

:
K
1/add_6Add1/mul_121/mul_13*
T0*
_output_shapes

:
И

1/Assign_6Assign1/Critic/target_net/l1/b11/add_6*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
O

1/mul_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
m
1/mul_14Mul
1/mul_14/x'1/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

1/mul_15/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
k
1/mul_15Mul
1/mul_15/x%1/Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
K
1/add_7Add1/mul_141/mul_15*
_output_shapes

:*
T0
Ъ

1/Assign_7Assign"1/Critic/target_net/q/dense/kernel1/add_7*
use_locking(*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
O

1/mul_16/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
g
1/mul_16Mul
1/mul_16/x%1/Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
O

1/mul_17/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
e
1/mul_17Mul
1/mul_17/x#1/Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
G
1/add_8Add1/mul_161/mul_17*
T0*
_output_shapes
:
Т

1/Assign_8Assign 1/Critic/target_net/q/dense/bias1/add_8*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
y
1/policy_grads/gradients/ShapeShape1/Actor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
g
"1/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
­
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ

?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
out_type0*
_output_shapes
:*
T0

A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Љ
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ

=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Е
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:џџџџџџџџџ

?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ш
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
й
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:

@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
§
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
й
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
б
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:

?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
р
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 

#1/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Ї
1/A_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape: 
к
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 

#1/A_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
Ї
1/A_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape: 
к
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

1/A_train/beta2_power/readIdentity1/A_train/beta2_power*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
П
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB*    *
dtype0
Ь
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:
Ѓ
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Н
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:
С
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB*    *
dtype0*
_output_shapes

:
Ю
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:
Љ
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
С
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:
Г
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueB*    *
dtype0*
_output_shapes
:
Р
'1/A_train/1/Actor/eval_net/l1/bias/Adam
VariableV2*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:

.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
Г
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
_output_shapes
:*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
Е
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueB*    
Т
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:

01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
З
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:
С
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
Ю
*1/A_train/1/Actor/eval_net/a/a/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container 
Ї
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Р
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*
_output_shapes

:*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
У
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB*    *
dtype0*
_output_shapes

:
а
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container 
­
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(
Ф
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:
Е
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    
Т
(1/A_train/1/Actor/eval_net/a/a/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:

/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ж
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
З
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    *
dtype0
Ф
*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ё
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
К
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
a
1/A_train/Adam/learning_rateConst*
valueB
 *oК*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
1/A_train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wЬ+2

:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0

81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:

;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:

91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( 

1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
Т
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(
К
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
 
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&Њќ} 0      Ѓ	Ш]хо­tзAJѓп
ѓ
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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

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
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.14.02v1.14.0-rc1-22-gaf24dc91b5О
f
S/sPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
f
R/rPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
h
S_/s_Placeholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
К
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
­
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*

seed*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes

:

80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:

40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:
Н
0/Actor/eval_net/l1/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container 
ў
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:

0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
_output_shapes

:*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Є
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
Б
0/Actor/eval_net/l1/bias
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ъ
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:

0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
_output_shapes
:*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Ђ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Њ
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
М
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB"      
Џ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
seed2
Ѓ
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:

50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:
П
0/Actor/eval_net/a/a/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container 

"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Ђ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:
І
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
Г
0/Actor/eval_net/a/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
ю
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:

0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
Й
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
­
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
w
0/Actor/eval_net/a/a/SigmoidSigmoid0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
b
0/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Sigmoid0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ
О
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB"      *
dtype0
Б
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *    
Г
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *ЭЬЬ=*
dtype0
Ё
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(*
dtype0
Ї
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:

60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:*
T0
С
0/Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape
:

#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
Ѕ
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:*
T0
Ј
,0/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@0/Actor/target_net/l1/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
Е
0/Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@0/Actor/target_net/l1/bias*
	container *
shape:
ђ
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
_output_shapes
:*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias
Ј
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
А
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB"      
Г
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Є
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed28
Ћ
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel

70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
У
0/Actor/target_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
	container *
shape
:

$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ј
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
_output_shapes

:*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
Њ
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
valueB*ЭЬЬ=
З
0/Actor/target_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@0/Actor/target_net/a/a/bias*
	container *
shape:
і
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:

 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
П
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Г
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
{
0/Actor/target_net/a/a/SigmoidSigmoid0/Actor/target_net/a/a/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Sigmoid0/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ
L
0/mul/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
a
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes

:*
T0
N
	0/mul_1/xConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
c
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:
E
0/addAdd0/mul0/mul_1*
T0*
_output_shapes

:
К
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:
N
	0/mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:
N
	0/mul_3/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
]
0/mul_3Mul	0/mul_3/x0/Actor/eval_net/l1/bias/read*
_output_shapes
:*
T0
E
0/add_1Add0/mul_20/mul_3*
_output_shapes
:*
T0
Ж

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
N
	0/mul_4/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
0/mul_4Mul	0/mul_4/x"0/Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:
N
	0/mul_5/xConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
d
0/mul_5Mul	0/mul_5/x 0/Actor/eval_net/a/a/kernel/read*
_output_shapes

:*
T0
I
0/add_2Add0/mul_40/mul_5*
_output_shapes

:*
T0
Р

0/Assign_2Assign0/Actor/target_net/a/a/kernel0/add_2*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
N
	0/mul_6/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
`
0/mul_6Mul	0/mul_6/x 0/Actor/target_net/a/a/bias/read*
_output_shapes
:*
T0
N
	0/mul_7/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
^
0/mul_7Mul	0/mul_7/x0/Actor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
E
0/add_3Add0/mul_60/mul_7*
T0*
_output_shapes
:
И

0/Assign_3Assign0/Actor/target_net/a/a/bias0/add_3*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
t
0/Critic/StopGradientStopGradient0/Actor/eval_net/a/scaled_a*
T0*'
_output_shapes
:џџџџџџџџџ
И
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ћ
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
­
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *ЭЬЬ=*
dtype0

H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2c*
dtype0*
_output_shapes

:

70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:

30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
Л
0/Critic/eval_net/l1/w1_s
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
њ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:

0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
И
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB"      
Ћ
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
­
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:*

seed

70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a

30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
Л
0/Critic/eval_net/l1/w1_a
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
њ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(

0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
Њ
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*ЭЬЬ=
З
0/Critic/eval_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape
:
ъ
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:

0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ђ
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Ж
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ

0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
T0

0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ц
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB"      
Й
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *    
Л
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
­
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2~*
dtype0*
_output_shapes

:*

seed
З
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
 
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Щ
 0/Critic/eval_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container *
shape
:

'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Б
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
А
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
Н
0/Critic/eval_net/q/dense/bias
VariableV2*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:

%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Ї
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ф
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
М
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB"      *
dtype0
Џ
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *ЭЬЬ=

J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2
Ѓ
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s

50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
П
0/Critic/target_net/l1/w1_s
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container 

"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(
Ђ
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
М
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB"      
Џ
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2
Ѓ
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a

50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
П
0/Critic/target_net/l1/w1_a
VariableV2*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0

"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ђ
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
Ў
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@0/Critic/target_net/l1/b1*
valueB*ЭЬЬ=
Л
0/Critic/target_net/l1/b1
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/target_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
ђ
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:

0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*,
_class"
 loc:@0/Critic/target_net/l1/b1*
_output_shapes

:*
T0
Ј
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Т
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 

0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
T0

0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*'
_output_shapes
:џџџџџџџџџ*
T0
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ъ
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Н
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
П
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *ЭЬЬ=
Д
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2Ј
П
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ј
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Э
"0/Critic/target_net/q/dense/kernel
VariableV2*
shared_name *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:

)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
З
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
Д
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
С
 0/Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
	container *
shape:

'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
­
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:
Ъ
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Т
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
U
0/target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
~
0/target_q/mulMul0/target_q/mul/x#0/Critic/target_net/q/dense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
\
0/target_q/addAddR/r0/target_q/mul*
T0*'
_output_shapes
:џџџџџџџџџ

0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
0/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
0/C_train/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0

60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ф
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0

.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
л
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

00/C_train/gradients/0/TD_error/Mean_grad/Shape_1Shape0/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
s
00/C_train/gradients/0/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
x
.0/C_train/gradients/0/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
-0/C_train/gradients/0/TD_error/Mean_grad/ProdProd00/C_train/gradients/0/TD_error/Mean_grad/Shape_1.0/C_train/gradients/0/TD_error/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
й
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
20/C_train/gradients/0/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
С
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
П
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ј
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
Ы
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
T0*
out_type0*
_output_shapes
:

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:

K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Д
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
т
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
и
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
ъ
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:џџџџџџџџџ*
T0

90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Г
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
ц
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
р
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
у
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
я
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
џ
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg
ї
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad

@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0

B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
к
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
є
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul
ё
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
ш
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*'
_output_shapes
:џџџџџџџџџ*
T0

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0

;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
њ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ї
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Ъ
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
о
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape
л
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:

70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0

90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0

G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
є
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
њ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ф
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
ж
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
м
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
ч
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
Ы
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
р
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
н
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:

=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
б
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ш
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
х
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1

#0/C_train/beta1_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ѕ
0/C_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: 
и
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(

0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 

#0/C_train/beta2_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ѕ
0/C_train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container 
и
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(

0/C_train/beta2_power/readIdentity0/C_train/beta2_power**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
Н
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes

:
Ъ
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:

/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
К
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
П
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0
Ь
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:
Ѕ
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
О
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
Н
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0
Ъ
(0/C_train/0/Critic/eval_net/l1/w1_a/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container 

/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
К
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
П
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Ь
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
Ѕ
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
О
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Й
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0
Ц
&0/C_train/0/Critic/eval_net/l1/b1/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container 

-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(
Д
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Л
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ш
(0/C_train/0/Critic/eval_net/l1/b1/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1

/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
И
-0/C_train/0/Critic/eval_net/l1/b1/Adam_1/readIdentity(0/C_train/0/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ы
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0
и
/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container 
Л
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Я
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
Э
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
к
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
С
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
г
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
П
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Ь
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:
Џ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Х
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
_output_shapes
:*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
С
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Ю
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:
Е
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
Щ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
0/C_train/Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Y
0/C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
0/C_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
0/C_train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wЬ+2
 
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( 
Ђ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a

70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( 
Ш
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 
Л
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
Ю
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Р
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
_output_shapes
: *
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(
а
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Ф
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
ў
0/C_train/AdamNoOp^0/C_train/Adam/Assign^0/C_train/Adam/Assign_18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam
y
0/a_grad/gradients/ShapeShape!0/Critic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
a
0/a_grad/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Љ
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
с
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
Ю
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
д
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0

:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ї
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
є
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0

60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:

80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:

F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ё
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ї
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
ъ
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
N
	0/mul_8/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
d
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	0/mul_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
b
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes

:
I
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes

:
М

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
O

0/mul_10/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
0/mul_10Mul
0/mul_10/x 0/Critic/target_net/l1/w1_a/read*
_output_shapes

:*
T0
O

0/mul_11/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
0/mul_11Mul
0/mul_11/x0/Critic/eval_net/l1/w1_a/read*
T0*
_output_shapes

:
K
0/add_5Add0/mul_100/mul_11*
T0*
_output_shapes

:
М

0/Assign_5Assign0/Critic/target_net/l1/w1_a0/add_5*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
O

0/mul_12/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
d
0/mul_12Mul
0/mul_12/x0/Critic/target_net/l1/b1/read*
T0*
_output_shapes

:
O

0/mul_13/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
0/mul_13Mul
0/mul_13/x0/Critic/eval_net/l1/b1/read*
_output_shapes

:*
T0
K
0/add_6Add0/mul_120/mul_13*
T0*
_output_shapes

:
И

0/Assign_6Assign0/Critic/target_net/l1/b10/add_6*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
O

0/mul_14/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
O

0/mul_15/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
k
0/mul_15Mul
0/mul_15/x%0/Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
K
0/add_7Add0/mul_140/mul_15*
T0*
_output_shapes

:
Ъ

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
O

0/mul_16/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
g
0/mul_16Mul
0/mul_16/x%0/Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
O

0/mul_17/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
e
0/mul_17Mul
0/mul_17/x#0/Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
G
0/add_8Add0/mul_160/mul_17*
T0*
_output_shapes
:
Т

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
out_type0*
_output_shapes
:*
T0
g
"0/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
­
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Sigmoid*
_output_shapes
:*
T0*
out_type0

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
Љ
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Д
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0

=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Е
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Sigmoid0/policy_grads/gradients/Fill*
T0*'
_output_shapes
:џџџџџџџџџ

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ш
F0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad0/Actor/eval_net/a/a/SigmoidA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
й
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
_output_shapes
:*
T0*
data_formatNHWC

@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 0/Actor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
§
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/TanhF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
й
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
б
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:

?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
р
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 

#0/A_train/beta1_power/initial_valueConst*
_output_shapes
: *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0
Ї
0/A_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
к
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(

0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias

#0/A_train/beta2_power/initial_valueConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ї
0/A_train/beta2_power
VariableV2*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
к
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(

0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
П
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes

:
Ь
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Ѓ
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(
Н
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:
С
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes

:
Ю
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Љ
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
С
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:
Г
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0
Р
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:

.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
Г
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:
Е
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *+
_class!
loc:@0/Actor/eval_net/l1/bias
Т
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:

00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
З
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:
С
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
Ю
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:
Ї
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Р
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:
У
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
а
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
­
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ф
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
_output_shapes

:*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Е
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Т
(0/A_train/0/Actor/eval_net/a/a/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:

/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ж
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
З
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Ф
*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ё
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
К
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:*
T0
a
0/A_train/Adam/learning_rateConst*
valueB
 *oК*
dtype0*
_output_shapes
: 
Y
0/A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
0/A_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
0/A_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0

:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:

80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0

;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0

90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 

0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Т
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
К
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
К
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
­
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2Т*
dtype0*
_output_shapes

:*

seed*
T0

81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel

41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:
Н
1/Actor/eval_net/l1/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel
ў
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel

1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:
Є
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
Б
1/Actor/eval_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:
ъ
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:

1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
_output_shapes
:*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
Ђ
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Њ
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
М
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB"      *
dtype0
Џ
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
seed2в*
dtype0
Ѓ
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:*
T0

51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:
П
1/Actor/eval_net/a/a/kernel
VariableV2*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 

"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:
Ђ
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:
І
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*ЭЬЬ=
Г
1/Actor/eval_net/a/a/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:
ю
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias

1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
Й
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
­
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
b
1/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
О
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *ЭЬЬ=
Ђ
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
seed2ф
Ї
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:

61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
С
1/Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container *
shape
:

#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:
Ѕ
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
_output_shapes

:*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
Ј
,1/Actor/target_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@1/Actor/target_net/l1/bias*
valueB*ЭЬЬ=
Е
1/Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@1/Actor/target_net/l1/bias*
	container *
shape:
ђ
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias

1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*-
_class#
!loc:@1/Actor/target_net/l1/bias*
_output_shapes
:*
T0
Ј
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
А
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB"      *
dtype0*
_output_shapes
:
Г
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ѕ
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed2є*
dtype0*
_output_shapes

:*

seed*
T0
Ћ
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:

71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
У
1/Actor/target_net/a/a/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container 

$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(
Ј
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:
Њ
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*
_output_shapes
:*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
valueB*ЭЬЬ=*
dtype0
З
1/Actor/target_net/a/a/bias
VariableV2*
shared_name *.
_class$
" loc:@1/Actor/target_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
і
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(

 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
_output_shapes
:*
T0
П
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Г
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
d
1/Actor/target_net/a/scaled_a/yConst*
valueB
 *  HC*
dtype0*
_output_shapes
: 

1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
L
1/mul/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
a
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
_output_shapes

:*
T0
N
	1/mul_1/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
c
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:
E
1/addAdd1/mul1/mul_1*
T0*
_output_shapes

:
К
1/AssignAssign1/Actor/target_net/l1/kernel1/add*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
N
	1/mul_2/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:
N
	1/mul_3/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
]
1/mul_3Mul	1/mul_3/x1/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:
E
1/add_1Add1/mul_21/mul_3*
T0*
_output_shapes
:
Ж

1/Assign_1Assign1/Actor/target_net/l1/bias1/add_1*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:
N
	1/mul_4/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
_output_shapes

:*
T0
N
	1/mul_5/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
1/mul_5Mul	1/mul_5/x 1/Actor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:
I
1/add_2Add1/mul_41/mul_5*
T0*
_output_shapes

:
Р

1/Assign_2Assign1/Actor/target_net/a/a/kernel1/add_2*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:
N
	1/mul_6/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
`
1/mul_6Mul	1/mul_6/x 1/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	1/mul_7/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
^
1/mul_7Mul	1/mul_7/x1/Actor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
E
1/add_3Add1/mul_61/mul_7*
_output_shapes
:*
T0
И

1/Assign_3Assign1/Actor/target_net/a/a/bias1/add_3*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
t
1/Critic/StopGradientStopGradient1/Actor/eval_net/a/scaled_a*
T0*'
_output_shapes
:џџџџџџџџџ
И
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ћ
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
­
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
seed2

71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:

31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Л
1/Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:
њ
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(

1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
И
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ћ
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
­
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
seed2Ј*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a

71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a

31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
Л
1/Critic/eval_net/l1/w1_a
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container 
њ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:

1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
Њ
)1/Critic/eval_net/l1/b1/Initializer/ConstConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*ЭЬЬ=*
dtype0*
_output_shapes

:
З
1/Critic/eval_net/l1/b1
VariableV2*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
ъ
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:

1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ђ
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Ж
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 

1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
T0

1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Й
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *ЭЬЬ=*
dtype0
Ў
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
seed2К
З
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
 
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Щ
 1/Critic/eval_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container *
shape
:

'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Б
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
А
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
_output_shapes
:*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*ЭЬЬ=*
dtype0
Н
1/Critic/eval_net/q/dense/bias
VariableV2*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Ї
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ф
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
М
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
М
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB"      *
dtype0
Џ
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
seed2Щ
Ѓ
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0

51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
П
1/Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
	container *
shape
:

"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
Ђ
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
М
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Џ
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *    
Б
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2в
Ѓ
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:

51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a
П
1/Critic/target_net/l1/w1_a
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
	container *
shape
:

"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ђ
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
Ў
+1/Critic/target_net/l1/b1/Initializer/ConstConst*,
_class"
 loc:@1/Critic/target_net/l1/b1*
valueB*ЭЬЬ=*
dtype0*
_output_shapes

:
Л
1/Critic/target_net/l1/b1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/target_net/l1/b1*
	container 
ђ
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1

1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
_output_shapes

:
Ј
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Т
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 

1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ

1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Н
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
П
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Д
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2ф
П
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ј
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
Э
"1/Critic/target_net/q/dense/kernel
VariableV2*
_output_shapes

:*
shared_name *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0

)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
З
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
Д
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
С
 1/Critic/target_net/q/dense/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
	container *
shape:

'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(
­
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias
Ъ
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Т
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
U
1/target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
~
1/target_q/mulMul1/target_q/mul/x#1/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
\
1/target_q/addAddR/r1/target_q/mul*
T0*'
_output_shapes
:џџџџџџџџџ

1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
a
1/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

1/TD_error/MeanMean1/TD_error/SquaredDifference1/TD_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
1/C_train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
b
1/C_train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 

61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ф
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0

.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
л
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ

01/C_train/gradients/1/TD_error/Mean_grad/Shape_1Shape1/TD_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
s
01/C_train/gradients/1/TD_error/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
x
.1/C_train/gradients/1/TD_error/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
е
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
й
/1/C_train/gradients/1/TD_error/Mean_grad/Prod_1Prod01/C_train/gradients/1/TD_error/Mean_grad/Shape_201/C_train/gradients/1/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
21/C_train/gradients/1/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
С
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
_output_shapes
: *
T0
П
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
Ј
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
Ы
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0

;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
T0*
out_type0*
_output_shapes
:

=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0

K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
т
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
и
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
ъ
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Г
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
ц
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
р
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
у
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
я
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
џ
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
ї
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad

@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(

B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
к
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
є
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
ё
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
ш
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:

;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
њ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
Ъ
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
о
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:

71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
out_type0*
_output_shapes
:*
T0

91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0

G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
є
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
њ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ф
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
ж
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape
м
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1

;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ч
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
Ы
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
р
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul
н
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:

=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
§
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
б
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ш
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
х
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:

#1/C_train/beta1_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ѕ
1/C_train/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container 
и
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 

1/C_train/beta1_power/readIdentity1/C_train/beta1_power**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0

#1/C_train/beta2_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ѕ
1/C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: 
и
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(

1/C_train/beta2_power/readIdentity1/C_train/beta2_power*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
Н
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Ъ
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:

/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(
К
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
П
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Ь
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
Ѕ
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(
О
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
Н
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
Ъ
(1/C_train/1/Critic/eval_net/l1/w1_a/Adam
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:

/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
К
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
П
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Ь
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
Ѕ
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
О
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Й
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ц
&1/C_train/1/Critic/eval_net/l1/b1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:

-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Д
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Л
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ш
(1/C_train/1/Critic/eval_net/l1/b1/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container 

/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/AssignAssign(1/C_train/1/Critic/eval_net/l1/b1/Adam_1:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
И
-1/C_train/1/Critic/eval_net/l1/b1/Adam_1/readIdentity(1/C_train/1/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ы
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
и
/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container 
Л
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Я
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
Э
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
к
11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
С
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(
г
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
П
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Ь
-1/C_train/1/Critic/eval_net/q/dense/bias/Adam
VariableV2*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Џ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(
Х
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
С
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Ю
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Е
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Щ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
1/C_train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o:
Y
1/C_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
1/C_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
 
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:
Ђ
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0

71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
Ш
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
Л
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
Ю
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
Р
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
а
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Ф
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
ў
1/C_train/AdamNoOp^1/C_train/Adam/Assign^1/C_train/Adam/Assign_18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam
y
1/a_grad/gradients/ShapeShape!1/Critic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
a
1/a_grad/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
Љ
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
с
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
Ю
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
д
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*'
_output_shapes
:џџџџџџџџџ*
T0

81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:

:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:

H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
є
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0

61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:

81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0

F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ё
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ї
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
њ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
ъ
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
N
	1/mul_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *Єp}?
d
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	1/mul_9/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
_output_shapes

:*
T0
I
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes

:
М

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
O

1/mul_10/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
f
1/mul_10Mul
1/mul_10/x 1/Critic/target_net/l1/w1_a/read*
T0*
_output_shapes

:
O

1/mul_11/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
1/mul_11Mul
1/mul_11/x1/Critic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
K
1/add_5Add1/mul_101/mul_11*
_output_shapes

:*
T0
М

1/Assign_5Assign1/Critic/target_net/l1/w1_a1/add_5*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
O

1/mul_12/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
d
1/mul_12Mul
1/mul_12/x1/Critic/target_net/l1/b1/read*
_output_shapes

:*
T0
O

1/mul_13/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
b
1/mul_13Mul
1/mul_13/x1/Critic/eval_net/l1/b1/read*
T0*
_output_shapes

:
K
1/add_6Add1/mul_121/mul_13*
_output_shapes

:*
T0
И

1/Assign_6Assign1/Critic/target_net/l1/b11/add_6*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
O

1/mul_14/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
m
1/mul_14Mul
1/mul_14/x'1/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

1/mul_15/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
k
1/mul_15Mul
1/mul_15/x%1/Critic/eval_net/q/dense/kernel/read*
T0*
_output_shapes

:
K
1/add_7Add1/mul_141/mul_15*
_output_shapes

:*
T0
Ъ

1/Assign_7Assign"1/Critic/target_net/q/dense/kernel1/add_7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel
O

1/mul_16/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
g
1/mul_16Mul
1/mul_16/x%1/Critic/target_net/q/dense/bias/read*
_output_shapes
:*
T0
O

1/mul_17/xConst*
_output_shapes
: *
valueB
 *
з#<*
dtype0
e
1/mul_17Mul
1/mul_17/x#1/Critic/eval_net/q/dense/bias/read*
T0*
_output_shapes
:
G
1/add_8Add1/mul_161/mul_17*
_output_shapes
:*
T0
Т

1/Assign_8Assign 1/Critic/target_net/q/dense/bias1/add_8*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias
y
1/policy_grads/gradients/ShapeShape1/Actor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
g
"1/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
­
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ

?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
_output_shapes
:*
T0*
out_type0

A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Љ
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Д
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ

=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Е
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:џџџџџџџџџ

?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ш
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
й
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:

@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
§
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
й
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
б
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:

?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
T0
р
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 

#1/A_train/beta1_power/initial_valueConst*
_output_shapes
: *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0
Ї
1/A_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias
к
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 

#1/A_train/beta2_power/initial_valueConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ї
1/A_train/beta2_power
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
к
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

1/A_train/beta2_power/readIdentity1/A_train/beta2_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
П
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes

:
Ь
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container 
Ѓ
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:
Н
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*
_output_shapes

:*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
С
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes

:
Ю
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Љ
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:
С
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes

:*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Г
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
valueB*    *+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
Р
'1/A_train/1/Actor/eval_net/l1/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0

.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
Г
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
_output_shapes
:*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
Е
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueB*    *+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:
Т
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:
З
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*
_output_shapes
:*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
С
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
Ю
*1/A_train/1/Actor/eval_net/a/a/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:
Ї
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Р
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*
_output_shapes

:*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
У
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:
а
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
­
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Ф
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:
Е
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Т
(1/A_train/1/Actor/eval_net/a/a/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:

/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ж
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
З
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Ф
*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:
Ё
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
К
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
a
1/A_train/Adam/learning_rateConst*
valueB
 *oК*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
[
1/A_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:

81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 

;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 

91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( 

1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
Т
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
Ц
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
К
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
 
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&"Х
trainable_variables­Њ

0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08

0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
Ђ
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08

0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08

0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08

0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08

0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
Ж
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
І
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08

1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08

1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
Ђ
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08

1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08

1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08

1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08

1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
Ж
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
І
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08"N
train_opB
@
0/C_train/Adam
0/A_train/Adam
1/C_train/Adam
1/A_train/Adam"­r
	variablesrr

0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08

0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
Ђ
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08

0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08
Є
0/Actor/target_net/l1/kernel:0#0/Actor/target_net/l1/kernel/Assign#0/Actor/target_net/l1/kernel/read:0280/Actor/target_net/l1/kernel/Initializer/random_normal:0

0/Actor/target_net/l1/bias:0!0/Actor/target_net/l1/bias/Assign!0/Actor/target_net/l1/bias/read:02.0/Actor/target_net/l1/bias/Initializer/Const:0
Ј
0/Actor/target_net/a/a/kernel:0$0/Actor/target_net/a/a/kernel/Assign$0/Actor/target_net/a/a/kernel/read:0290/Actor/target_net/a/a/kernel/Initializer/random_normal:0

0/Actor/target_net/a/a/bias:0"0/Actor/target_net/a/a/bias/Assign"0/Actor/target_net/a/a/bias/read:02/0/Actor/target_net/a/a/bias/Initializer/Const:0

0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08

0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08

0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
Ж
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
І
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08
 
0/Critic/target_net/l1/w1_s:0"0/Critic/target_net/l1/w1_s/Assign"0/Critic/target_net/l1/w1_s/read:0270/Critic/target_net/l1/w1_s/Initializer/random_normal:0
 
0/Critic/target_net/l1/w1_a:0"0/Critic/target_net/l1/w1_a/Assign"0/Critic/target_net/l1/w1_a/read:0270/Critic/target_net/l1/w1_a/Initializer/random_normal:0

0/Critic/target_net/l1/b1:0 0/Critic/target_net/l1/b1/Assign 0/Critic/target_net/l1/b1/read:02-0/Critic/target_net/l1/b1/Initializer/Const:0
М
$0/Critic/target_net/q/dense/kernel:0)0/Critic/target_net/q/dense/kernel/Assign)0/Critic/target_net/q/dense/kernel/read:02>0/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
Ќ
"0/Critic/target_net/q/dense/bias:0'0/Critic/target_net/q/dense/bias/Assign'0/Critic/target_net/q/dense/bias/read:0240/Critic/target_net/q/dense/bias/Initializer/Const:0
|
0/C_train/beta1_power:00/C_train/beta1_power/Assign0/C_train/beta1_power/read:02%0/C_train/beta1_power/initial_value:0
|
0/C_train/beta2_power:00/C_train/beta2_power/Assign0/C_train/beta2_power/read:02%0/C_train/beta2_power/initial_value:0
Ь
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
д
,0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
Ь
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
д
,0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
Ф
(0/C_train/0/Critic/eval_net/l1/b1/Adam:0-0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign-0/C_train/0/Critic/eval_net/l1/b1/Adam/read:02:0/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
Ь
*0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/read:02<0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
ш
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam:060/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/read:02C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
№
30/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1:080/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/read:02E0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
р
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam:040/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign40/C_train/0/Critic/eval_net/q/dense/bias/Adam/read:02A0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
ш
10/C_train/0/Critic/eval_net/q/dense/bias/Adam_1:060/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/read:02C0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
0/A_train/beta1_power:00/A_train/beta1_power/Assign0/A_train/beta1_power/read:02%0/A_train/beta1_power/initial_value:0
|
0/A_train/beta2_power:00/A_train/beta2_power/Assign0/A_train/beta2_power/read:02%0/A_train/beta2_power/initial_value:0
а
+0/A_train/0/Actor/eval_net/l1/kernel/Adam:000/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign00/A_train/0/Actor/eval_net/l1/kernel/Adam/read:02=0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
и
-0/A_train/0/Actor/eval_net/l1/kernel/Adam_1:020/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/read:02?0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
Ш
)0/A_train/0/Actor/eval_net/l1/bias/Adam:0.0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign.0/A_train/0/Actor/eval_net/l1/bias/Adam/read:02;0/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
а
+0/A_train/0/Actor/eval_net/l1/bias/Adam_1:000/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign00/A_train/0/Actor/eval_net/l1/bias/Adam_1/read:02=0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
д
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam:010/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign10/A_train/0/Actor/eval_net/a/a/kernel/Adam/read:02>0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
м
.0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1:030/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/read:02@0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
Ь
*0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign/0/A_train/0/Actor/eval_net/a/a/bias/Adam/read:02<0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
д
,0/A_train/0/Actor/eval_net/a/a/bias/Adam_1:010/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/read:02>0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0

1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08

1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
Ђ
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08

1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08
Є
1/Actor/target_net/l1/kernel:0#1/Actor/target_net/l1/kernel/Assign#1/Actor/target_net/l1/kernel/read:0281/Actor/target_net/l1/kernel/Initializer/random_normal:0

1/Actor/target_net/l1/bias:0!1/Actor/target_net/l1/bias/Assign!1/Actor/target_net/l1/bias/read:02.1/Actor/target_net/l1/bias/Initializer/Const:0
Ј
1/Actor/target_net/a/a/kernel:0$1/Actor/target_net/a/a/kernel/Assign$1/Actor/target_net/a/a/kernel/read:0291/Actor/target_net/a/a/kernel/Initializer/random_normal:0

1/Actor/target_net/a/a/bias:0"1/Actor/target_net/a/a/bias/Assign"1/Actor/target_net/a/a/bias/read:02/1/Actor/target_net/a/a/bias/Initializer/Const:0

1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08

1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08

1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
Ж
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
І
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08
 
1/Critic/target_net/l1/w1_s:0"1/Critic/target_net/l1/w1_s/Assign"1/Critic/target_net/l1/w1_s/read:0271/Critic/target_net/l1/w1_s/Initializer/random_normal:0
 
1/Critic/target_net/l1/w1_a:0"1/Critic/target_net/l1/w1_a/Assign"1/Critic/target_net/l1/w1_a/read:0271/Critic/target_net/l1/w1_a/Initializer/random_normal:0

1/Critic/target_net/l1/b1:0 1/Critic/target_net/l1/b1/Assign 1/Critic/target_net/l1/b1/read:02-1/Critic/target_net/l1/b1/Initializer/Const:0
М
$1/Critic/target_net/q/dense/kernel:0)1/Critic/target_net/q/dense/kernel/Assign)1/Critic/target_net/q/dense/kernel/read:02>1/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
Ќ
"1/Critic/target_net/q/dense/bias:0'1/Critic/target_net/q/dense/bias/Assign'1/Critic/target_net/q/dense/bias/read:0241/Critic/target_net/q/dense/bias/Initializer/Const:0
|
1/C_train/beta1_power:01/C_train/beta1_power/Assign1/C_train/beta1_power/read:02%1/C_train/beta1_power/initial_value:0
|
1/C_train/beta2_power:01/C_train/beta2_power/Assign1/C_train/beta2_power/read:02%1/C_train/beta2_power/initial_value:0
Ь
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
д
,1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
Ь
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
д
,1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
Ф
(1/C_train/1/Critic/eval_net/l1/b1/Adam:0-1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign-1/C_train/1/Critic/eval_net/l1/b1/Adam/read:02:1/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
Ь
*1/C_train/1/Critic/eval_net/l1/b1/Adam_1:0/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/read:02<1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
ш
11/C_train/1/Critic/eval_net/q/dense/kernel/Adam:061/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/read:02C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
№
31/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1:081/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/read:02E1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
р
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam:041/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign41/C_train/1/Critic/eval_net/q/dense/bias/Adam/read:02A1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
ш
11/C_train/1/Critic/eval_net/q/dense/bias/Adam_1:061/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/read:02C1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
1/A_train/beta1_power:01/A_train/beta1_power/Assign1/A_train/beta1_power/read:02%1/A_train/beta1_power/initial_value:0
|
1/A_train/beta2_power:01/A_train/beta2_power/Assign1/A_train/beta2_power/read:02%1/A_train/beta2_power/initial_value:0
а
+1/A_train/1/Actor/eval_net/l1/kernel/Adam:001/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign01/A_train/1/Actor/eval_net/l1/kernel/Adam/read:02=1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
и
-1/A_train/1/Actor/eval_net/l1/kernel/Adam_1:021/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/read:02?1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
Ш
)1/A_train/1/Actor/eval_net/l1/bias/Adam:0.1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign.1/A_train/1/Actor/eval_net/l1/bias/Adam/read:02;1/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
а
+1/A_train/1/Actor/eval_net/l1/bias/Adam_1:001/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign01/A_train/1/Actor/eval_net/l1/bias/Adam_1/read:02=1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
д
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam:011/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign11/A_train/1/Actor/eval_net/a/a/kernel/Adam/read:02>1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
м
.1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1:031/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/read:02@1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
Ь
*1/A_train/1/Actor/eval_net/a/a/bias/Adam:0/1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign/1/A_train/1/Actor/eval_net/a/a/bias/Adam/read:02<1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
д
,1/A_train/1/Actor/eval_net/a/a/bias/Adam_1:011/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/read:02>1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:06Н