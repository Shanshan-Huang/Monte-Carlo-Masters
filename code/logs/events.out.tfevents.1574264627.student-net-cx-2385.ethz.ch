       БK"	  └╠WuОAbrain.Event:2^ЄХ7у     ░8&	укп╠WuОA"ф╬
f
S/sPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
f
R/rPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
h
S_/s_Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
║
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Г
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
»
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Џ
I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes

:d*

seed*
T0
Ъ
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
ѕ
40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
й
0/Actor/eval_net/l1/kernel
VariableV2*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
■
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
Ъ
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
ц
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueBd*  ђ?*+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
▒
0/Actor/eval_net/l1/bias
VariableV2*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
Ж
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Ћ
0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
б
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b( 
ф
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:         d*
T0
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*'
_output_shapes
:         d*
T0
╝
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
»
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
▒
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
ъ
J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Б
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
ї
50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
┐
0/Actor/eval_net/a/a/kernel
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container 
ѓ
"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
б
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
д
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*  ђ?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
│
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
Ь
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ў
0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
_output_shapes
:*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
╣
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Г
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
w
0/Actor/eval_net/a/a/SigmoidSigmoid0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
b
0/Actor/eval_net/a/scaled_a/yConst*
_output_shapes
: *
valueB
 *  ╚B*
dtype0
Љ
0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Sigmoid0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
Й
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"   d   */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0
▒
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
│
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
А
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(
Д
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d
љ
60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d
┴
0/Actor/target_net/l1/kernel
VariableV2*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
є
#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
Ц
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d
е
,0/Actor/target_net/l1/bias/Initializer/ConstConst*
valueBd*  ђ?*-
_class#
!loc:@0/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:d
х
0/Actor/target_net/l1/bias
VariableV2*-
_class#
!loc:@0/Actor/target_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
Ы
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
Џ
0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:d
е
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b( 
░
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:         d*
T0
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*'
_output_shapes
:         d*
T0
└
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
│
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *═╠L=*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0
х
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
ц
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed28*
dtype0*
_output_shapes

:d*

seed*
T0
Ф
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
ћ
70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
├
0/Actor/target_net/a/a/kernel
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
і
$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
е
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
ф
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*  ђ?*.
_class$
" loc:@0/Actor/target_net/a/a/bias
и
0/Actor/target_net/a/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@0/Actor/target_net/a/a/bias
Ш
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
ъ
 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
┐
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
│
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
{
0/Actor/target_net/a/a/SigmoidSigmoid0/Actor/target_net/a/a/BiasAdd*'
_output_shapes
:         *
T0
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Ќ
0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Sigmoid0/Actor/target_net/a/scaled_a/y*'
_output_shapes
:         *
T0
L
0/mul/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
a
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes

:d*
T0
N
	0/mul_1/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
c
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:d
E
0/addAdd0/mul0/mul_1*
T0*
_output_shapes

:d
║
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d
N
	0/mul_2/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
_output_shapes
:d*
T0
N
	0/mul_3/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
]
0/mul_3Mul	0/mul_3/x0/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:d
E
0/add_1Add0/mul_20/mul_3*
T0*
_output_shapes
:d
Х

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias
N
	0/mul_4/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
f
0/mul_4Mul	0/mul_4/x"0/Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:d
N
	0/mul_5/xConst*
_output_shapes
: *
valueB
 *
О#<*
dtype0
d
0/mul_5Mul	0/mul_5/x 0/Actor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:d
I
0/add_2Add0/mul_40/mul_5*
T0*
_output_shapes

:d
└

0/Assign_2Assign0/Actor/target_net/a/a/kernel0/add_2*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
N
	0/mul_6/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
`
0/mul_6Mul	0/mul_6/x 0/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	0/mul_7/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
^
0/mul_7Mul	0/mul_7/x0/Actor/eval_net/a/a/bias/read*
T0*
_output_shapes
:
E
0/add_3Add0/mul_60/mul_7*
_output_shapes
:*
T0
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
0/Critic/StopGradientStopGradient0/Actor/eval_net/a/scaled_a*'
_output_shapes
:         *
T0
И
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
Ф
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Г
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0
ў
H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2c
Џ
70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
ё
30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
╗
0/Critic/eval_net/l1/w1_s
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
Щ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
ю
0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
И
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ф
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
Г
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
ў
H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:
Џ
70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
ё
30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
╗
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
Щ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
ю
0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
ф
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*  ђ?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
и
0/Critic/eval_net/l1/b1
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
Ж
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
ќ
0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
б
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Х
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Ї
0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:         *
T0
І
0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:         
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:         
к
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
╣
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
╗
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Г
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2~*
dtype0*
_output_shapes

:*

seed
и
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
а
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
╔
 0/Critic/eval_net/q/dense/kernel
VariableV2*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
ќ
'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
▒
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
░
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*  ђ?*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
й
0/Critic/eval_net/q/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container 
ѓ
%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Д
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
─
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
╝
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
╝
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
»
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
▒
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
Ъ
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2Ї*
dtype0
Б
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0
ї
50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
┐
0/Critic/target_net/l1/w1_s
VariableV2*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape
:*
dtype0
ѓ
"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
б
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
╝
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
»
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
▒
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Ъ
J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2ќ*
dtype0*
_output_shapes

:
Б
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
ї
50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
┐
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
ѓ
"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
б
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
«
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*  ђ?*,
_class"
 loc:@0/Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
╗
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
Ы
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
ю
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
е
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
┬
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
Њ
0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:         
Љ
0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:         
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:         
╩
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
й
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
┐
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
┤
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2е
┐
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
е
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
═
"0/Critic/target_net/q/dense/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
ъ
)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
и
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
┤
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*  ђ?*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
┴
 0/Critic/target_net/q/dense/bias
VariableV2*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
і
'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(
Г
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:
╩
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
┬
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
U
0/target_q/mul/xConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
~
0/target_q/mulMul0/target_q/mul/x#0/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:         
\
0/target_q/addAddR/r0/target_q/mul*
T0*'
_output_shapes
:         
ќ
0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:         
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ё
0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
0/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
0/C_train/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ї
0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Є
60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
─
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
і
.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
█
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
ї
00/C_train/gradients/0/TD_error/Mean_grad/Shape_1Shape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
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
Н
-0/C_train/gradients/0/TD_error/Mean_grad/ProdProd00/C_train/gradients/0/TD_error/Mean_grad/Shape_1.0/C_train/gradients/0/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
┘
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
┴
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
_output_shapes
: *
T0
┐
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
е
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
╦
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*'
_output_shapes
:         *
T0
Ѕ
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
_output_shapes
:*
T0*
out_type0
ъ
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ю
K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
Р
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
п
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
Ж
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         
і
90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ђ
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ј
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
│
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
╩
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
Т
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:         
Я
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         
с
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
№
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
 
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg
э
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad
ъ
@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
І
B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
┌
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
З
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:         *
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul
ы
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
У
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*'
_output_shapes
:         *
T0
Љ
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
ї
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ќ
I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
є
70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Щ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
і
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
э
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
╩
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
я
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:         
█
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:
њ
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
ќ
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
Љ
G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Њ
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ќ
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Щ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
─
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
о
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:         
▄
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:         
Ѕ
;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
у
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
╦
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
Я
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:         *
T0
П
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:
Ї
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
Л
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
У
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:         *
T0
т
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
ћ
#0/C_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ц
0/C_train/beta1_power
VariableV2*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: 
п
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
і
0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
ћ
#0/C_train/beta2_power/initial_valueConst*
valueB
 *wЙ?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ц
0/C_train/beta2_power
VariableV2*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0
п
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(
і
0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
й
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB*    
╩
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
Ъ
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(
║
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
┐
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
_output_shapes

:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0
╠
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:
Ц
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Й
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
й
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
╩
(0/C_train/0/Critic/eval_net/l1/w1_a/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Ъ
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
║
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
┐
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
╠
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
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
Ц
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
Й
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
╣
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0
к
&0/C_train/0/Critic/eval_net/l1/b1/Adam
VariableV2*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
Ќ
-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
┤
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
╗
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
╚
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
Ю
/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
И
-0/C_train/0/Critic/eval_net/l1/b1/Adam_1/readIdentity(0/C_train/0/Critic/eval_net/l1/b1/Adam_1*
_output_shapes

:*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
╦
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
п
/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam
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
╗
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
¤
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
═
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
┌
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1
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
┴
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
М
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
┐
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
╠
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
»
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
┼
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
┴
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
╬
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
х
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╔
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
0/C_train/Adam/learning_rateConst*
valueB
 *иЛ8*
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
 *wЙ?*
dtype0*
_output_shapes
: 
[
0/C_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
а
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:*
use_locking( 
б
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
Ћ
70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
╚
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
╗
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
use_nesterov( 
╬
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
└
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
л
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
─
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
■
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
 *  ђ?*
dtype0*
_output_shapes
: 
Џ
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Е
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
р
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
╬
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
н
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
љ
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
І
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB"      *
dtype0
ћ
H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ѓ
60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
э
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Є
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0
Љ
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Ћ
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
ј
F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
 
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ѓ
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
э
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Щ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
Ж
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
N
	0/mul_8/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
d
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	0/mul_9/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
b
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes

:
I
0/add_4Add0/mul_80/mul_9*
_output_shapes

:*
T0
╝

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(
O

0/mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *цp}?
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
О#<*
dtype0*
_output_shapes
: 
d
0/mul_11Mul
0/mul_11/x0/Critic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
K
0/add_5Add0/mul_100/mul_11*
_output_shapes

:*
T0
╝

0/Assign_5Assign0/Critic/target_net/l1/w1_a0/add_5*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
O

0/mul_12/xConst*
valueB
 *цp}?*
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

0/mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
О#<
b
0/mul_13Mul
0/mul_13/x0/Critic/eval_net/l1/b1/read*
T0*
_output_shapes

:
K
0/add_6Add0/mul_120/mul_13*
_output_shapes

:*
T0
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

0/mul_14/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

0/mul_15/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
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
╩

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
O

0/mul_16/xConst*
valueB
 *цp}?*
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
О#<*
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
┬

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
out_type0*
_output_shapes
:*
T0
g
"0/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Г
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Џ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Sigmoid*
_output_shapes
:*
T0*
out_type0
ё
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Е
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┤
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
ћ
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
х
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Sigmoid0/policy_grads/gradients/Fill*'
_output_shapes
:         *
T0
џ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ђ
C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
У
F0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad0/Actor/eval_net/a/a/SigmoidA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape*'
_output_shapes
:         *
T0
┘
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
ї
@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 0/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:         d*
transpose_a( *
transpose_b(*
T0
§
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/TanhF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes

:d*
transpose_a(*
transpose_b( 
┘
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:         d
Л
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Ѓ
?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
Я
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
_output_shapes

:d*
transpose_a(*
transpose_b( 
ќ
#0/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Д
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
┌
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ї
0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
ќ
#0/A_train/beta2_power/initial_valueConst*
valueB
 *wЙ?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Д
0/A_train/beta2_power
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
┌
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ї
0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
╦
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
х
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
й
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:d*
T0
╠
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Б
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
_output_shapes

:d*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(
й
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
═
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
и
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    
├
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:d*
T0
╬
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Е
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
_output_shapes

:d*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(
┴
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes

:d*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
│
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
_output_shapes
:d*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    *
dtype0
└
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
Ќ
.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
│
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
х
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    *
dtype0*
_output_shapes
:d
┬
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d
Ю
00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
и
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0
┴
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
╬
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
VariableV2*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
Д
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
└
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
├
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
л
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d
Г
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
─
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
х
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    
┬
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
Џ
/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Х
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
и
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
─
*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1
VariableV2*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
А
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
║
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
a
0/A_train/Adam/learning_rateConst*
valueB
 *иЛИ*
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
 *wЙ?*
dtype0*
_output_shapes
: 
[
0/A_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Ќ
:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:d
Ї
80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d
Ю
;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d
Њ
90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
ї
0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
┬
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ј
0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
к
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
║
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
║
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Г
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
»
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
ю
I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2к*
dtype0*
_output_shapes

:d*

seed*
T0
Ъ
81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
ѕ
41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
й
1/Actor/eval_net/l1/kernel
VariableV2*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
■
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
Ъ
1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
ц
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueBd*  ђ?*+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
▒
1/Actor/eval_net/l1/bias
VariableV2*+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
Ж
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(
Ћ
1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
б
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b( 
ф
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         d
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:         d
╝
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"d      *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0
»
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
▒
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0
Ъ
J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
seed2о
Б
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
ї
51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
┐
1/Actor/eval_net/a/a/kernel
VariableV2*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
ѓ
"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
б
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
д
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*  ђ?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0
│
1/Actor/eval_net/a/a/bias
VariableV2*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ь
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
ў
1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
╣
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Г
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
b
1/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Љ
1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
Й
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"   d   */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
▒
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
│
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
б
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
_output_shapes

:d*

seed*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
seed2У*
dtype0
Д
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes

:d*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
љ
61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:d
┴
1/Actor/target_net/l1/kernel
VariableV2*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
є
#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d
Ц
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:d
е
,1/Actor/target_net/l1/bias/Initializer/ConstConst*
valueBd*  ђ?*-
_class#
!loc:@1/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:d
х
1/Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@1/Actor/target_net/l1/bias*
	container *
shape:d
Ы
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
Џ
1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*
_output_shapes
:d*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias
е
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b( 
░
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         d
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:         d
└
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"d      *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0
│
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *═╠L=*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
х
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Ц
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed2Э*
dtype0*
_output_shapes

:d*

seed
Ф
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
ћ
71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
├
1/Actor/target_net/a/a/kernel
VariableV2*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
і
$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
е
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0
ф
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*  ђ?*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
и
1/Actor/target_net/a/a/bias
VariableV2*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ш
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
ъ
 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
_output_shapes
:*
T0
┐
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
│
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*'
_output_shapes
:         *
T0
d
1/Actor/target_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Ќ
1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:         
L
1/mul/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
a
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
T0*
_output_shapes

:d
N
	1/mul_1/xConst*
_output_shapes
: *
valueB
 *
О#<*
dtype0
c
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:d
E
1/addAdd1/mul1/mul_1*
T0*
_output_shapes

:d
║
1/AssignAssign1/Actor/target_net/l1/kernel1/add*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d
N
	1/mul_2/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
_output_shapes
:d*
T0
N
	1/mul_3/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
]
1/mul_3Mul	1/mul_3/x1/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:d
E
1/add_1Add1/mul_21/mul_3*
T0*
_output_shapes
:d
Х

1/Assign_1Assign1/Actor/target_net/l1/bias1/add_1*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(
N
	1/mul_4/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:d
N
	1/mul_5/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
d
1/mul_5Mul	1/mul_5/x 1/Actor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:d
I
1/add_2Add1/mul_41/mul_5*
_output_shapes

:d*
T0
└

1/Assign_2Assign1/Actor/target_net/a/a/kernel1/add_2*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(
N
	1/mul_6/xConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0
`
1/mul_6Mul	1/mul_6/x 1/Actor/target_net/a/a/bias/read*
_output_shapes
:*
T0
N
	1/mul_7/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
^
1/mul_7Mul	1/mul_7/x1/Actor/eval_net/a/a/bias/read*
_output_shapes
:*
T0
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
:         
И
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
Ф
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Г
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Ў
H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
seed2Б*
dtype0*
_output_shapes

:*

seed
Џ
71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
ё
31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
╗
1/Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:
Щ
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
ю
1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
И
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ф
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
Г
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0
Ў
H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
seed2г
Џ
71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
ё
31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
╗
1/Critic/eval_net/l1/w1_a
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
Щ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
ю
1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
ф
)1/Critic/eval_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*
valueB*  ђ?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0
и
1/Critic/eval_net/l1/b1
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
Ж
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
ќ
1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
б
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Х
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
Ї
1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:         
І
1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:         
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:         
к
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
╣
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
╗
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
«
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
seed2Й*
dtype0*
_output_shapes

:*

seed
и
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
а
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
╔
 1/Critic/eval_net/q/dense/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
ќ
'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
▒
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
░
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*  ђ?*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
й
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
ѓ
%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Д
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
─
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
╝
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
╝
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
»
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0
▒
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
Ъ
J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
seed2═*
dtype0*
_output_shapes

:*

seed
Б
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
ї
51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
┐
1/Critic/target_net/l1/w1_s
VariableV2*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
ѓ
"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
б
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
╝
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0
»
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
▒
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*.
_class$
" loc:@1/Critic/target_net/l1/w1_a
Ъ
J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2о
Б
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
ї
51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
┐
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
ѓ
"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
б
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
«
+1/Critic/target_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*
valueB*  ђ?*,
_class"
 loc:@1/Critic/target_net/l1/b1*
dtype0
╗
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
Ы
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
ю
1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
_output_shapes

:
е
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
┬
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
Њ
1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*'
_output_shapes
:         *
T0
Љ
1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:         
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:         
╩
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"      *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0
й
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
┐
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
┤
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2У*
dtype0*
_output_shapes

:*

seed*
T0
┐
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
е
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
═
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
ъ
)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
и
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
┤
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*  ђ?*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
dtype0
┴
 1/Critic/target_net/q/dense/bias
VariableV2*
shared_name *3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
і
'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Г
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias
╩
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
┬
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
U
1/target_q/mul/xConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
~
1/target_q/mulMul1/target_q/mul/x#1/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:         
\
1/target_q/addAddR/r1/target_q/mul*
T0*'
_output_shapes
:         
ќ
1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:         *
T0
a
1/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ё
1/TD_error/MeanMean1/TD_error/SquaredDifference1/TD_error/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
1/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
1/C_train/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ї
1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Є
61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
─
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
і
.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
█
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*
T0*'
_output_shapes
:         *

Tmultiples0
ї
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
Н
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
┘
/1/C_train/gradients/1/TD_error/Mean_grad/Prod_1Prod01/C_train/gradients/1/TD_error/Mean_grad/Shape_201/C_train/gradients/1/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
21/C_train/gradients/1/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
┴
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
┐
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
е
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
╦
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:         
Ѕ
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
T0*
out_type0*
_output_shapes
:
ъ
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ю
K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Р
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
п
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
Ж
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:         *
T0
і
91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ђ
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
ј
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
│
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
╩
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
Т
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         *
T0*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
Я
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         
с
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
№
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
 
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         
э
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ъ
@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:         *
transpose_a( 
І
B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
┌
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
З
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:         
ы
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
У
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
Љ
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
ї
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ќ
I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
є
71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Щ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
і
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
э
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
╩
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
я
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape
█
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
њ
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
out_type0*
_output_shapes
:*
T0
ќ
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0
Љ
G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Њ
51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Ќ
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Щ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
─
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
о
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:         
▄
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
Ѕ
;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
у
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
╦
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
Я
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:         
П
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
Ї
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
§
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Л
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
У
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:         
т
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
_output_shapes

:*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ћ
#1/C_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ц
1/C_train/beta1_power
VariableV2**
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
п
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(
і
1/C_train/beta1_power/readIdentity1/C_train/beta1_power*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
ћ
#1/C_train/beta2_power/initial_valueConst*
valueB
 *wЙ?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
Ц
1/C_train/beta2_power
VariableV2**
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
п
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
і
1/C_train/beta2_power/readIdentity1/C_train/beta2_power*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
й
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0*
_output_shapes

:
╩
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
Ъ
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
║
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
┐
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB*    *
dtype0*
_output_shapes

:
╠
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Ц
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Й
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
й
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
╩
(1/C_train/1/Critic/eval_net/l1/w1_a/Adam
VariableV2*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ъ
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
║
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
┐
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
╠
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ц
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Й
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
╣
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
к
&1/C_train/1/Critic/eval_net/l1/b1/Adam
VariableV2**
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ќ
-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
┤
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:*
T0
╗
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
╚
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
Ю
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
╦
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0
п
/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam
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
╗
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
¤
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
═
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
┌
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
┴
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
М
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
┐
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    
╠
-1/C_train/1/Critic/eval_net/q/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:
»
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
┼
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
┴
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
╬
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
х
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╔
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
1/C_train/Adam/learning_rateConst*
valueB
 *иЛ8*
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
 *wЙ?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
а
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:
б
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Ћ
71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
╚
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
╗
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
╬
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
└
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
л
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
─
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
■
1/C_train/AdamNoOp^1/C_train/Adam/Assign^1/C_train/Adam/Assign_18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam
y
1/a_grad/gradients/ShapeShape!1/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
a
1/a_grad/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Џ
1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Е
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
_output_shapes
:*
T0*
data_formatNHWC
р
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b(*
T0
╬
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
н
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
љ
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
І
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
ћ
H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ѓ
61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
э
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Є
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Љ
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Ћ
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
ј
F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
 
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ы
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Ѓ
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
э
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
Щ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
Ж
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
N
	1/mul_8/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
d
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
_output_shapes

:*
T0
N
	1/mul_9/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
b
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
_output_shapes

:*
T0
I
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes

:
╝

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
O

1/mul_10/xConst*
valueB
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
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
╝

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
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
b
1/mul_13Mul
1/mul_13/x1/Critic/eval_net/l1/b1/read*
_output_shapes

:*
T0
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

1/mul_14/xConst*
valueB
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
k
1/mul_15Mul
1/mul_15/x%1/Critic/eval_net/q/dense/kernel/read*
_output_shapes

:*
T0
K
1/add_7Add1/mul_141/mul_15*
T0*
_output_shapes

:
╩

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
 *цp}?*
dtype0*
_output_shapes
: 
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
О#<*
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
┬

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
"1/policy_grads/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
Г
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Џ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
out_type0*
_output_shapes
:*
T0
ё
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Е
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
ћ
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
х
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:         
џ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ђ
C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
У
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:         
┘
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
ї
@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:         d*
transpose_a( *
transpose_b(*
T0
§
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes

:d*
transpose_a(*
transpose_b( 
┘
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*'
_output_shapes
:         d*
T0
Л
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Ѓ
?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b(
Я
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
transpose_b( *
T0*
_output_shapes

:d*
transpose_a(
ќ
#1/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Д
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
┌
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
ї
1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
_output_shapes
: *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
ќ
#1/A_train/beta2_power/initial_valueConst*
valueB
 *wЙ?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
Д
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
┌
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
ї
1/A_train/beta2_power/readIdentity1/A_train/beta2_power*
_output_shapes
: *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
╦
K1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"   d   
х
A1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
й
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes

:d
╠
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
Б
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d
й
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
═
M1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
и
C1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
├
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:d*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*

index_type0
╬
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
Е
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
_output_shapes

:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(
┴
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
│
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*    *
dtype0*
_output_shapes
:d
└
'1/A_train/1/Actor/eval_net/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d
Ќ
.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
│
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
_output_shapes
:d*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
х
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:d*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*    
┬
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d
Ю
01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
и
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0
┴
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
╬
*1/A_train/1/Actor/eval_net/a/a/kernel/Adam
VariableV2*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
Д
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
└
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
├
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
л
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container 
Г
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
─
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
х
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
┬
(1/A_train/1/Actor/eval_net/a/a/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:
Џ
/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Х
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
и
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
─
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
А
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
║
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
_output_shapes
:*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
a
1/A_train/Adam/learning_rateConst*
valueB
 *иЛИ*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Y
1/A_train/Adam/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
[
1/A_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Ќ
:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes

:d*
use_locking( *
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Ї
81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
use_locking( *
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( 
Ю
;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d
Њ
91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
ї
1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
┬
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
ј
1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
к
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(
║
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
а
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&Е&а 8     KаЁМ	YЦ█╠WuОAJЊ­
џз
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
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
Ї
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
2	љ
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
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ё
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
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

2	љ
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
ї
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
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ*1.14.02v1.14.0-rc1-22-gaf24dc91b5ф╬
f
S/sPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
f
R/rPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
h
S_/s_Placeholder*'
_output_shapes
:         *
shape:         *
dtype0
║
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Г
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
»
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Џ
I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2
Ъ
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
ѕ
40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
й
0/Actor/eval_net/l1/kernel
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container 
■
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d
Ъ
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
ц
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*  ђ?*
dtype0*
_output_shapes
:d
▒
0/Actor/eval_net/l1/bias
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias
Ж
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ћ
0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
б
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:         d*
transpose_b( *
T0
ф
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         d
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*'
_output_shapes
:         d*
T0
╝
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
»
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
▒
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
ъ
J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
seed2*
dtype0*
_output_shapes

:d*

seed*
T0
Б
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
ї
50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
┐
0/Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d
ѓ
"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
б
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
д
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*
_output_shapes
:*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*  ђ?*
dtype0
│
0/Actor/eval_net/a/a/bias
VariableV2*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ь
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ў
0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
╣
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
Г
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
w
0/Actor/eval_net/a/a/SigmoidSigmoid0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
b
0/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Љ
0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Sigmoid0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
Й
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
▒
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
_output_shapes
: */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *═╠L=*
dtype0
│
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
А
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(*
dtype0*
_output_shapes

:d
Д
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d
љ
60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d*
T0
┴
0/Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes

:d*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape
:d
є
#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d
Ц
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes

:d
е
,0/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@0/Actor/target_net/l1/bias*
valueBd*  ђ?*
dtype0*
_output_shapes
:d
х
0/Actor/target_net/l1/bias
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/target_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
Ы
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
Џ
0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:d
е
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:         d*
transpose_b( *
T0
░
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         d
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*'
_output_shapes
:         d*
T0
└
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
│
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
х
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
ц
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed28
Ф
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0
ћ
70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
├
0/Actor/target_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
	container *
shape
:d
і
$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
е
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
ф
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
_output_shapes
:*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
valueB*  ђ?*
dtype0
и
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
Ш
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
ъ
 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
┐
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
│
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
{
0/Actor/target_net/a/a/SigmoidSigmoid0/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Ќ
0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Sigmoid0/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:         
L
0/mul/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
a
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes

:d*
T0
N
	0/mul_1/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
c
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:d
E
0/addAdd0/mul0/mul_1*
T0*
_output_shapes

:d
║
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
N
	0/mul_2/xConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
_output_shapes
:d*
T0
N
	0/mul_3/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
]
0/mul_3Mul	0/mul_3/x0/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:d
E
0/add_1Add0/mul_20/mul_3*
T0*
_output_shapes
:d
Х

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
N
	0/mul_4/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
f
0/mul_4Mul	0/mul_4/x"0/Actor/target_net/a/a/kernel/read*
_output_shapes

:d*
T0
N
	0/mul_5/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
d
0/mul_5Mul	0/mul_5/x 0/Actor/eval_net/a/a/kernel/read*
_output_shapes

:d*
T0
I
0/add_2Add0/mul_40/mul_5*
T0*
_output_shapes

:d
└

0/Assign_2Assign0/Actor/target_net/a/a/kernel0/add_2*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
N
	0/mul_6/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
`
0/mul_6Mul	0/mul_6/x 0/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	0/mul_7/xConst*
_output_shapes
: *
valueB
 *
О#<*
dtype0
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
0/Critic/StopGradientStopGradient0/Actor/eval_net/a/scaled_a*'
_output_shapes
:         *
T0
И
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ф
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
ў
H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
seed2c*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Џ
70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:
ё
30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
╗
0/Critic/eval_net/l1/w1_s
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Щ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
ю
0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
И
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ф
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0
Г
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *   ?*
dtype0
ў
H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
seed2l*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Џ
70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
ё
30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
╗
0/Critic/eval_net/l1/w1_a
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
Щ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
ю
0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
ф
)0/Critic/eval_net/l1/b1/Initializer/ConstConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*  ђ?*
dtype0*
_output_shapes

:
и
0/Critic/eval_net/l1/b1
VariableV2**
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ж
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
ќ
0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
б
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
Х
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
Ї
0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:         *
T0
І
0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*'
_output_shapes
:         *
T0
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:         
к
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
╣
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *   ?
Г
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2~
и
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
а
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
╔
 0/Critic/eval_net/q/dense/kernel
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
ќ
'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
▒
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
░
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*  ђ?*
dtype0*
_output_shapes
:
й
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
ѓ
%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Д
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
─
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
╝
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
╝
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB"      *
dtype0
»
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ъ
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2Ї
Б
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:
ї
50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0
┐
0/Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape
:
ѓ
"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
б
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes

:*
T0
╝
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
»
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ъ
J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2ќ
Б
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
ї
50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
┐
0/Critic/target_net/l1/w1_a
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_a
ѓ
"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
б
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
«
+0/Critic/target_net/l1/b1/Initializer/ConstConst*,
_class"
 loc:@0/Critic/target_net/l1/b1*
valueB*  ђ?*
dtype0*
_output_shapes

:
╗
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
Ы
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
ю
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
_output_shapes

:
е
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
┬
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
Њ
0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:         
Љ
0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:         
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:         
╩
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0
й
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
┤
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2е
┐
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
е
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
═
"0/Critic/target_net/q/dense/kernel
VariableV2*
dtype0*
_output_shapes

:*
shared_name *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:
ъ
)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
и
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
┤
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
valueB*  ђ?*
dtype0*
_output_shapes
:
┴
 0/Critic/target_net/q/dense/bias
VariableV2*
shared_name *3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
і
'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias
Г
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:
╩
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b( *
T0
┬
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
U
0/target_q/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
~
0/target_q/mulMul0/target_q/mul/x#0/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:         
\
0/target_q/addAddR/r0/target_q/mul*
T0*'
_output_shapes
:         
ќ
0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:         *
T0
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ё
0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
 *  ђ?*
dtype0
Ї
0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Є
60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
─
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
і
.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
█
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
ї
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
Н
-0/C_train/gradients/0/TD_error/Mean_grad/ProdProd00/C_train/gradients/0/TD_error/Mean_grad/Shape_1.0/C_train/gradients/0/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
┘
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
20/C_train/gradients/0/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
┴
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
┐
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
е
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
╦
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:         
Ѕ
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
T0*
out_type0*
_output_shapes
:
ъ
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ю
K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Р
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
п
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
Ж
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         
і
90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ђ
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
ј
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
є
?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
│
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
╩
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
Т
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:         
Я
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         *
T0
с
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
№
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
 
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         *
T0
э
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ъ
@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
І
B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
┌
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
З
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:         
ы
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
У
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
Љ
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
ї
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ќ
I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
є
70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Щ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
і
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
э
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
╩
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
я
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:         
█
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:
њ
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
ќ
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
Љ
G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Њ
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
З
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Ќ
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Щ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
─
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
о
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:         
▄
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
Ѕ
;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:         *
transpose_b(*
T0
у
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
╦
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
Я
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:         *
T0
П
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
Ї
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
Л
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
У
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:         
т
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
ћ
#0/C_train/beta1_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ц
0/C_train/beta1_power
VariableV2*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: 
п
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
і
0/C_train/beta1_power/readIdentity0/C_train/beta1_power**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
ћ
#0/C_train/beta2_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Ц
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
п
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
і
0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
й
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes

:
╩
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ъ
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
║
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
┐
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes

:
╠
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape
:*
dtype0
Ц
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
Й
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
й
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
╩
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
Ъ
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
║
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
┐
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
╠
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
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
Ц
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(
Й
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
╣
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0
к
&0/C_train/0/Critic/eval_net/l1/b1/Adam
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
Ќ
-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
┤
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
_output_shapes

:*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
╗
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0
╚
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
Ю
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
╦
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
п
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
╗
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
¤
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
═
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
┌
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1
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
┴
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
М
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
┐
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
╠
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container 
»
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
┼
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
┴
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
╬
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
х
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
╔
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
0/C_train/Adam/learning_rateConst*
valueB
 *иЛ8*
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
0/C_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *wЙ?*
dtype0
[
0/C_train/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w╠+2*
dtype0
а
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:
б
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
use_nesterov( 
Ћ
70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( 
╚
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
╗
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
╬
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
└
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
л
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
─
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
■
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
 *  ђ?*
dtype0*
_output_shapes
: 
Џ
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*

index_type0*'
_output_shapes
:         *
T0
Е
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
р
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
╬
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
н
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
љ
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
І
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
ћ
H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ѓ
60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Є
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
З
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Љ
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Ћ
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0
ј
F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
 
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ѓ
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Щ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:         *
transpose_b(*
T0
Ж
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
N
	0/mul_8/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
d
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	0/mul_9/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
b
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
_output_shapes

:*
T0
I
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes

:
╝

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
O

0/mul_10/xConst*
valueB
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
d
0/mul_11Mul
0/mul_11/x0/Critic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
K
0/add_5Add0/mul_100/mul_11*
_output_shapes

:*
T0
╝

0/Assign_5Assign0/Critic/target_net/l1/w1_a0/add_5*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(
O

0/mul_12/xConst*
valueB
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
b
0/mul_13Mul
0/mul_13/x0/Critic/eval_net/l1/b1/read*
T0*
_output_shapes

:
K
0/add_6Add0/mul_120/mul_13*
_output_shapes

:*
T0
И

0/Assign_6Assign0/Critic/target_net/l1/b10/add_6*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(
O

0/mul_14/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

0/mul_15/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
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
╩

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
O

0/mul_16/xConst*
valueB
 *цp}?*
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
О#<*
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
┬

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
g
"0/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Г
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*'
_output_shapes
:         *
T0*

index_type0
Џ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Sigmoid*
T0*
out_type0*
_output_shapes
:
ё
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Е
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:         *
T0
ћ
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
х
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Sigmoid0/policy_grads/gradients/Fill*
T0*'
_output_shapes
:         
џ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
У
F0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad0/Actor/eval_net/a/a/SigmoidA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:         
┘
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
ї
@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 0/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         d*
transpose_b(
§
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/TanhF0/policy_grads/gradients/0/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
transpose_a(*
_output_shapes

:d*
transpose_b( *
T0
┘
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:         d
Л
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Ѓ
?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
Я
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes

:d*
transpose_b( 
ќ
#0/A_train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *fff?
Д
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
┌
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ї
0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
ќ
#0/A_train/beta2_power/initial_valueConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Д
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
┌
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ї
0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
╦
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
х
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0
й
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
╠
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
Б
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
й
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
═
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
и
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
├
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
╬
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
Е
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d
┴
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes

:d
│
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
_output_shapes
:d*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0
└
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias
Ќ
.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
│
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
х
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0
┬
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d
Ю
00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
и
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
┴
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
╬
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Д
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
└
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
├
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
л
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name 
Г
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(
─
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
х
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0
┬
(0/A_train/0/Actor/eval_net/a/a/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container 
Џ
/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
Х
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
_output_shapes
:*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
и
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
─
*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1
VariableV2*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
А
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
║
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
a
0/A_train/Adam/learning_rateConst*
valueB
 *иЛИ*
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
 *wЙ?*
dtype0*
_output_shapes
: 
[
0/A_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Ќ
:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes

:d*
use_locking( *
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Ї
80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
_output_shapes
:d*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Ю
;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d*
use_locking( *
T0
Њ
90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
ї
0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
┬
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
ј
0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
к
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
║
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
║
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Г
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *═╠L=
»
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *   ?
ю
I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2к*
dtype0*
_output_shapes

:d*

seed
Ъ
81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
ѕ
41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
й
1/Actor/eval_net/l1/kernel
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
■
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d
Ъ
1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
ц
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*
_output_shapes
:d*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*  ђ?*
dtype0
▒
1/Actor/eval_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d
Ж
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ћ
1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
б
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         d*
transpose_b( 
ф
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         d
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:         d
╝
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
»
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
▒
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ъ
J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
seed2о
Б
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
ї
51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
┐
1/Actor/eval_net/a/a/kernel
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
ѓ
"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
б
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
д
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*  ђ?*
dtype0*
_output_shapes
:
│
1/Actor/eval_net/a/a/bias
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ь
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(
ў
1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
╣
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
Г
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
b
1/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Љ
1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
Й
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB"   d   
▒
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
│
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
б
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
seed2У*
dtype0*
_output_shapes

:d*

seed*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
Д
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:d
љ
61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes

:d
┴
1/Actor/target_net/l1/kernel
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container 
є
#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d
Ц
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
_output_shapes

:d*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
е
,1/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@1/Actor/target_net/l1/bias*
valueBd*  ђ?*
dtype0*
_output_shapes
:d
х
1/Actor/target_net/l1/bias
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@1/Actor/target_net/l1/bias*
	container *
shape:d
Ы
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(
Џ
1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*
_output_shapes
:d*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias
е
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:         d*
transpose_b( *
T0
░
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:         d*
T0
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*'
_output_shapes
:         d*
T0
└
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
│
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *═╠L=*
dtype0*
_output_shapes
: 
х
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *   ?
Ц
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed2Э
Ф
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:d*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
ћ
71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
├
1/Actor/target_net/a/a/kernel
VariableV2*
shared_name *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
і
$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
е
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
ф
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
valueB*  ђ?*
dtype0*
_output_shapes
:
и
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
Ш
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(
ъ
 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*
_output_shapes
:*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias
┐
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
│
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:         
d
1/Actor/target_net/a/scaled_a/yConst*
valueB
 *  ╚B*
dtype0*
_output_shapes
: 
Ќ
1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:         
L
1/mul/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
a
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
T0*
_output_shapes

:d
N
	1/mul_1/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
c
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes

:d
E
1/addAdd1/mul1/mul_1*
T0*
_output_shapes

:d
║
1/AssignAssign1/Actor/target_net/l1/kernel1/add*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
N
	1/mul_2/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
_output_shapes
:d*
T0
N
	1/mul_3/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
]
1/mul_3Mul	1/mul_3/x1/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:d
E
1/add_1Add1/mul_21/mul_3*
T0*
_output_shapes
:d
Х

1/Assign_1Assign1/Actor/target_net/l1/bias1/add_1*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias
N
	1/mul_4/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
_output_shapes

:d*
T0
N
	1/mul_5/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
d
1/mul_5Mul	1/mul_5/x 1/Actor/eval_net/a/a/kernel/read*
_output_shapes

:d*
T0
I
1/add_2Add1/mul_41/mul_5*
T0*
_output_shapes

:d
└

1/Assign_2Assign1/Actor/target_net/a/a/kernel1/add_2*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
N
	1/mul_6/xConst*
valueB
 *цp}?*
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
О#<*
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
1/Critic/StopGradientStopGradient1/Actor/eval_net/a/scaled_a*'
_output_shapes
:         *
T0
И
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB"      *
dtype0*
_output_shapes
:
Ф
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ў
H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
seed2Б
Џ
71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
ё
31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
╗
1/Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:
Щ
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
ю
1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
И
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ф
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *    
Г
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ў
H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
seed2г
Џ
71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
ё
31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
╗
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
Щ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
ю
1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
ф
)1/Critic/eval_net/l1/b1/Initializer/ConstConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*  ђ?*
dtype0*
_output_shapes

:
и
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
Ж
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
ќ
1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
б
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
Х
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
Ї
1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:         *
T0
І
1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:         
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*'
_output_shapes
:         *
T0
к
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
╣
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
«
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
seed2Й
и
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
а
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
╔
 1/Critic/eval_net/q/dense/kernel
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
ќ
'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(
▒
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
░
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*  ђ?*
dtype0*
_output_shapes
:
й
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
ѓ
%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
Д
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
─
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
╝
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
╝
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB"      *
dtype0*
_output_shapes
:
»
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ъ
J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
seed2═
Б
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
ї
51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s
┐
1/Critic/target_net/l1/w1_s
VariableV2*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
	container *
shape
:*
dtype0*
_output_shapes

:
ѓ
"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
б
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes

:
╝
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB"      
»
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *   ?*
dtype0
Ъ
J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2о*
dtype0*
_output_shapes

:
Б
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
ї
51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
┐
1/Critic/target_net/l1/w1_a
VariableV2*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
ѓ
"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
б
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
«
+1/Critic/target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@1/Critic/target_net/l1/b1*
valueB*  ђ?
╗
1/Critic/target_net/l1/b1
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/target_net/l1/b1*
	container *
shape
:*
dtype0
Ы
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
ю
1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
_output_shapes

:
е
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b( 
┬
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:         
Њ
1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:         
Љ
1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*'
_output_shapes
:         *
T0
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:         
╩
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
й
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *    
┐
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
┤
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2У*
dtype0*
_output_shapes

:*

seed
┐
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
е
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
═
"1/Critic/target_net/q/dense/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel
ъ
)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
и
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
┤
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*
_output_shapes
:*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
valueB*  ђ?*
dtype0
┴
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
і
'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Г
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
_output_shapes
:
╩
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:         
┬
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
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
:         *
T0
\
1/target_q/addAddR/r1/target_q/mul*
T0*'
_output_shapes
:         
ќ
1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:         *
T0
a
1/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ё
1/TD_error/MeanMean1/TD_error/SquaredDifference1/TD_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
\
1/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
1/C_train/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ї
1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Є
61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
─
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
і
.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
█
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
ї
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
.1/C_train/gradients/1/TD_error/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Н
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
┘
/1/C_train/gradients/1/TD_error/Mean_grad/Prod_1Prod01/C_train/gradients/1/TD_error/Mean_grad/Shape_201/C_train/gradients/1/TD_error/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
21/C_train/gradients/1/TD_error/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
┴
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
┐
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
е
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
╦
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*'
_output_shapes
:         *
T0
Ѕ
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
T0*
out_type0*
_output_shapes
:
ъ
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ю
K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
Р
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
п
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:         
Ж
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:         
і
91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ђ
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ј
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
є
?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
│
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:         
╩
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
Т
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:         *
T0
Я
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:         *
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg
с
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
№
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
 
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:         
э
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ъ
@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b(*
T0
І
B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
┌
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
З
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:         
ы
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
У
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
Љ
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
ї
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
Ќ
I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
є
71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Щ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
і
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
╩
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
я
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:         
█
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
њ
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
ќ
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
Љ
G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Њ
51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
З
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
Ќ
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Щ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
─
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
о
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:         
▄
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
Ѕ
;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:         *
transpose_b(*
T0
у
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
╦
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
Я
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:         
П
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:
Ї
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
§
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
Л
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
У
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:         
т
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
ћ
#1/C_train/beta1_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ц
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
п
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
і
1/C_train/beta1_power/readIdentity1/C_train/beta1_power**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
ћ
#1/C_train/beta2_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Ц
1/C_train/beta2_power
VariableV2**
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
п
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
і
1/C_train/beta2_power/readIdentity1/C_train/beta2_power**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
й
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
╩
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Ъ
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
║
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:*
T0
┐
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes

:
╠
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape
:
Ц
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes

:
Й
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes

:
й
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
╩
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
Ъ
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
║
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
┐
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
╠
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1
VariableV2*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ц
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Й
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
╣
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
к
&1/C_train/1/Critic/eval_net/l1/b1/Adam
VariableV2**
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ќ
-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
┤
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:*
T0
╗
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1
╚
(1/C_train/1/Critic/eval_net/l1/b1/Adam_1
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
Ю
/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/AssignAssign(1/C_train/1/Critic/eval_net/l1/b1/Adam_1:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(
И
-1/C_train/1/Critic/eval_net/l1/b1/Adam_1/readIdentity(1/C_train/1/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
╦
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
п
/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam
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
╗
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
¤
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
═
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
┌
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
┴
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
М
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
┐
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
╠
-1/C_train/1/Critic/eval_net/q/dense/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container 
»
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
┼
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
┴
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0
╬
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1
VariableV2*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
х
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
╔
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
1/C_train/Adam/learning_rateConst*
valueB
 *иЛ8*
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
 *wЙ?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
а
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes

:*
use_locking( 
б
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
Ћ
71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
╚
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
╗
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
╬
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
└
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
л
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
─
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
■
1/C_train/AdamNoOp^1/C_train/Adam/Assign^1/C_train/Adam/Assign_18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam
y
1/a_grad/gradients/ShapeShape!1/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
a
1/a_grad/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
Џ
1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Е
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
р
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:         *
transpose_b(*
T0
╬
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
н
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:         
љ
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
І
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
ћ
H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ѓ
61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
Є
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
З
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
Љ
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Ћ
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0
ј
F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
 
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ѓ
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
э
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Щ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
Ж
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
N
	1/mul_8/xConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: 
d
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes

:
N
	1/mul_9/xConst*
_output_shapes
: *
valueB
 *
О#<*
dtype0
b
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes

:
I
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes

:
╝

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes

:
O

1/mul_10/xConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0
f
1/mul_10Mul
1/mul_10/x 1/Critic/target_net/l1/w1_a/read*
_output_shapes

:*
T0
O

1/mul_11/xConst*
valueB
 *
О#<*
dtype0*
_output_shapes
: 
d
1/mul_11Mul
1/mul_11/x1/Critic/eval_net/l1/w1_a/read*
T0*
_output_shapes

:
K
1/add_5Add1/mul_101/mul_11*
_output_shapes

:*
T0
╝

1/Assign_5Assign1/Critic/target_net/l1/w1_a1/add_5*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
O

1/mul_12/xConst*
valueB
 *цp}?*
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
О#<*
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

1/Assign_6Assign1/Critic/target_net/l1/b11/add_6*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(
O

1/mul_14/xConst*
valueB
 *цp}?*
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
О#<*
dtype0*
_output_shapes
: 
k
1/mul_15Mul
1/mul_15/x%1/Critic/eval_net/q/dense/kernel/read*
_output_shapes

:*
T0
K
1/add_7Add1/mul_141/mul_15*
_output_shapes

:*
T0
╩

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
 *цp}?*
dtype0*
_output_shapes
: 
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
О#<*
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
┬

1/Assign_8Assign 1/Critic/target_net/q/dense/bias1/add_8*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
y
1/policy_grads/gradients/ShapeShape1/Actor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
g
"1/policy_grads/gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Г
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:         
Џ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
T0*
out_type0*
_output_shapes
:
ё
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Е
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┤
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:         
ћ
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ї
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
х
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:         
џ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
У
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*
T0*'
_output_shapes
:         
┘
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
ї
@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         d*
transpose_b(
§
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
transpose_a(*
_output_shapes

:d*
transpose_b( *
T0
┘
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:         d
Л
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Ѓ
?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:         *
transpose_b(
Я
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
transpose_a(*
_output_shapes

:d*
transpose_b( *
T0
ќ
#1/A_train/beta1_power/initial_valueConst*
_output_shapes
: *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0
Д
1/A_train/beta1_power
VariableV2*
_output_shapes
: *
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0
┌
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ї
1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
ќ
#1/A_train/beta2_power/initial_valueConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Д
1/A_train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias
┌
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
ї
1/A_train/beta2_power/readIdentity1/A_train/beta2_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
╦
K1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
х
A1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel
й
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*

index_type0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d*
T0
╠
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
Б
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
_output_shapes

:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(
й
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*
_output_shapes

:d*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
═
M1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
и
C1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
├
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
╬
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container 
Е
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes

:d
┴
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes

:d
│
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*
valueBd*    *+
_class!
loc:@1/Actor/eval_net/l1/bias
└
'1/A_train/1/Actor/eval_net/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d
Ќ
.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
│
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
х
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueBd*    *+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
┬
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias
Ю
01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
и
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0
┴
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:d*
valueBd*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
╬
*1/A_train/1/Actor/eval_net/a/a/kernel/Adam
VariableV2*
	container *
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
Д
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
└
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
├
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
л
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
Г
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
─
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
х
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
┬
(1/A_train/1/Actor/eval_net/a/a/bias/Adam
VariableV2*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Џ
/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Х
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
_output_shapes
:*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
и
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
─
*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1
VariableV2*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
А
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
║
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
a
1/A_train/Adam/learning_rateConst*
valueB
 *иЛИ*
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
1/A_train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЙ?
[
1/A_train/Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
Ќ
:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes

:d*
use_locking( 
Ї
81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d
Ю
;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d
Њ
91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
ї
1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
┬
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
ј
1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
к
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
║
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
а
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&"┼
trainable_variablesГф
ъ
0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08
ј
0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
б
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
њ
0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08
џ
0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
џ
0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
і
0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
Х
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
д
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08
ъ
1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08
ј
1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
б
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
њ
1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08
џ
1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
џ
1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
і
1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
Х
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
д
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08"N
train_opB
@
0/C_train/Adam
0/A_train/Adam
1/C_train/Adam
1/A_train/Adam"Гr
	variablesЪrюr
ъ
0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08
ј
0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
б
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
њ
0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08
ц
0/Actor/target_net/l1/kernel:0#0/Actor/target_net/l1/kernel/Assign#0/Actor/target_net/l1/kernel/read:0280/Actor/target_net/l1/kernel/Initializer/random_normal:0
ћ
0/Actor/target_net/l1/bias:0!0/Actor/target_net/l1/bias/Assign!0/Actor/target_net/l1/bias/read:02.0/Actor/target_net/l1/bias/Initializer/Const:0
е
0/Actor/target_net/a/a/kernel:0$0/Actor/target_net/a/a/kernel/Assign$0/Actor/target_net/a/a/kernel/read:0290/Actor/target_net/a/a/kernel/Initializer/random_normal:0
ў
0/Actor/target_net/a/a/bias:0"0/Actor/target_net/a/a/bias/Assign"0/Actor/target_net/a/a/bias/read:02/0/Actor/target_net/a/a/bias/Initializer/Const:0
џ
0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
џ
0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
і
0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
Х
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
д
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08
а
0/Critic/target_net/l1/w1_s:0"0/Critic/target_net/l1/w1_s/Assign"0/Critic/target_net/l1/w1_s/read:0270/Critic/target_net/l1/w1_s/Initializer/random_normal:0
а
0/Critic/target_net/l1/w1_a:0"0/Critic/target_net/l1/w1_a/Assign"0/Critic/target_net/l1/w1_a/read:0270/Critic/target_net/l1/w1_a/Initializer/random_normal:0
љ
0/Critic/target_net/l1/b1:0 0/Critic/target_net/l1/b1/Assign 0/Critic/target_net/l1/b1/read:02-0/Critic/target_net/l1/b1/Initializer/Const:0
╝
$0/Critic/target_net/q/dense/kernel:0)0/Critic/target_net/q/dense/kernel/Assign)0/Critic/target_net/q/dense/kernel/read:02>0/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
г
"0/Critic/target_net/q/dense/bias:0'0/Critic/target_net/q/dense/bias/Assign'0/Critic/target_net/q/dense/bias/read:0240/Critic/target_net/q/dense/bias/Initializer/Const:0
|
0/C_train/beta1_power:00/C_train/beta1_power/Assign0/C_train/beta1_power/read:02%0/C_train/beta1_power/initial_value:0
|
0/C_train/beta2_power:00/C_train/beta2_power/Assign0/C_train/beta2_power/read:02%0/C_train/beta2_power/initial_value:0
╠
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
н
,0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
╠
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
н
,0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
─
(0/C_train/0/Critic/eval_net/l1/b1/Adam:0-0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign-0/C_train/0/Critic/eval_net/l1/b1/Adam/read:02:0/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
╠
*0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/read:02<0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
У
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam:060/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/read:02C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
­
30/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1:080/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/read:02E0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
Я
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam:040/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign40/C_train/0/Critic/eval_net/q/dense/bias/Adam/read:02A0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
У
10/C_train/0/Critic/eval_net/q/dense/bias/Adam_1:060/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/read:02C0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
0/A_train/beta1_power:00/A_train/beta1_power/Assign0/A_train/beta1_power/read:02%0/A_train/beta1_power/initial_value:0
|
0/A_train/beta2_power:00/A_train/beta2_power/Assign0/A_train/beta2_power/read:02%0/A_train/beta2_power/initial_value:0
л
+0/A_train/0/Actor/eval_net/l1/kernel/Adam:000/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign00/A_train/0/Actor/eval_net/l1/kernel/Adam/read:02=0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
п
-0/A_train/0/Actor/eval_net/l1/kernel/Adam_1:020/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/read:02?0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
╚
)0/A_train/0/Actor/eval_net/l1/bias/Adam:0.0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign.0/A_train/0/Actor/eval_net/l1/bias/Adam/read:02;0/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
л
+0/A_train/0/Actor/eval_net/l1/bias/Adam_1:000/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign00/A_train/0/Actor/eval_net/l1/bias/Adam_1/read:02=0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
н
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam:010/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign10/A_train/0/Actor/eval_net/a/a/kernel/Adam/read:02>0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
▄
.0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1:030/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/read:02@0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
╠
*0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign/0/A_train/0/Actor/eval_net/a/a/bias/Adam/read:02<0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
н
,0/A_train/0/Actor/eval_net/a/a/bias/Adam_1:010/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/read:02>0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0
ъ
1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08
ј
1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
б
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
њ
1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08
ц
1/Actor/target_net/l1/kernel:0#1/Actor/target_net/l1/kernel/Assign#1/Actor/target_net/l1/kernel/read:0281/Actor/target_net/l1/kernel/Initializer/random_normal:0
ћ
1/Actor/target_net/l1/bias:0!1/Actor/target_net/l1/bias/Assign!1/Actor/target_net/l1/bias/read:02.1/Actor/target_net/l1/bias/Initializer/Const:0
е
1/Actor/target_net/a/a/kernel:0$1/Actor/target_net/a/a/kernel/Assign$1/Actor/target_net/a/a/kernel/read:0291/Actor/target_net/a/a/kernel/Initializer/random_normal:0
ў
1/Actor/target_net/a/a/bias:0"1/Actor/target_net/a/a/bias/Assign"1/Actor/target_net/a/a/bias/read:02/1/Actor/target_net/a/a/bias/Initializer/Const:0
џ
1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
џ
1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
і
1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
Х
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
д
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08
а
1/Critic/target_net/l1/w1_s:0"1/Critic/target_net/l1/w1_s/Assign"1/Critic/target_net/l1/w1_s/read:0271/Critic/target_net/l1/w1_s/Initializer/random_normal:0
а
1/Critic/target_net/l1/w1_a:0"1/Critic/target_net/l1/w1_a/Assign"1/Critic/target_net/l1/w1_a/read:0271/Critic/target_net/l1/w1_a/Initializer/random_normal:0
љ
1/Critic/target_net/l1/b1:0 1/Critic/target_net/l1/b1/Assign 1/Critic/target_net/l1/b1/read:02-1/Critic/target_net/l1/b1/Initializer/Const:0
╝
$1/Critic/target_net/q/dense/kernel:0)1/Critic/target_net/q/dense/kernel/Assign)1/Critic/target_net/q/dense/kernel/read:02>1/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
г
"1/Critic/target_net/q/dense/bias:0'1/Critic/target_net/q/dense/bias/Assign'1/Critic/target_net/q/dense/bias/read:0241/Critic/target_net/q/dense/bias/Initializer/Const:0
|
1/C_train/beta1_power:01/C_train/beta1_power/Assign1/C_train/beta1_power/read:02%1/C_train/beta1_power/initial_value:0
|
1/C_train/beta2_power:01/C_train/beta2_power/Assign1/C_train/beta2_power/read:02%1/C_train/beta2_power/initial_value:0
╠
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
н
,1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
╠
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
н
,1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
─
(1/C_train/1/Critic/eval_net/l1/b1/Adam:0-1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign-1/C_train/1/Critic/eval_net/l1/b1/Adam/read:02:1/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
╠
*1/C_train/1/Critic/eval_net/l1/b1/Adam_1:0/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/read:02<1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
У
11/C_train/1/Critic/eval_net/q/dense/kernel/Adam:061/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/read:02C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
­
31/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1:081/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/read:02E1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
Я
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam:041/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign41/C_train/1/Critic/eval_net/q/dense/bias/Adam/read:02A1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
У
11/C_train/1/Critic/eval_net/q/dense/bias/Adam_1:061/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/read:02C1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
1/A_train/beta1_power:01/A_train/beta1_power/Assign1/A_train/beta1_power/read:02%1/A_train/beta1_power/initial_value:0
|
1/A_train/beta2_power:01/A_train/beta2_power/Assign1/A_train/beta2_power/read:02%1/A_train/beta2_power/initial_value:0
л
+1/A_train/1/Actor/eval_net/l1/kernel/Adam:001/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign01/A_train/1/Actor/eval_net/l1/kernel/Adam/read:02=1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
п
-1/A_train/1/Actor/eval_net/l1/kernel/Adam_1:021/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/read:02?1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
╚
)1/A_train/1/Actor/eval_net/l1/bias/Adam:0.1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign.1/A_train/1/Actor/eval_net/l1/bias/Adam/read:02;1/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
л
+1/A_train/1/Actor/eval_net/l1/bias/Adam_1:001/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign01/A_train/1/Actor/eval_net/l1/bias/Adam_1/read:02=1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
н
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam:011/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign11/A_train/1/Actor/eval_net/a/a/kernel/Adam/read:02>1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
▄
.1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1:031/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/read:02@1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
╠
*1/A_train/1/Actor/eval_net/a/a/bias/Adam:0/1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign/1/A_train/1/Actor/eval_net/a/a/bias/Adam/read:02<1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
н
,1/A_train/1/Actor/eval_net/a/a/bias/Adam_1:011/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/read:02>1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0Xq╝