       ЃK"	  у[uзAbrain.Event:2Тн­КWx     3Ъм	хЖЃу[uзA"Ъ№
h
S/sPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
f
R/rPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
j
S_/s_Placeholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
К
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
­
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *
з#<*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Џ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 

I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes
:	d*

seed*
T0
 
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d

40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
П
0/Actor/eval_net/l1/kernel
VariableV2*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
џ
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(
 
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Є
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:d*
valueBd*  ?*+
_class!
loc:@0/Actor/eval_net/l1/bias
Б
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
ъ
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d

0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Ђ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b( 
Њ
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*'
_output_shapes
:џџџџџџџџџd*
T0
М
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
Џ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *
з#<*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
Б
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 

J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
seed2*
dtype0*
_output_shapes

:d*

seed
Ѓ
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d

50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
П
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

"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
Ђ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
І
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*  ?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
Г
0/Actor/eval_net/a/a/bias
VariableV2*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0
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
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
­
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
q
0/Actor/eval_net/a/a/ReluRelu0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
b
0/Actor/eval_net/a/scaled_a/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Relu0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:џџџџџџџџџ
О
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"  d   */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
Б
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *
з#<*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
Г
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0
Ђ
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	d*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(
Ј
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d

60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d
У
0/Actor/target_net/l1/kernel
VariableV2*
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container 

#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
І
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d
Ј
,0/Actor/target_net/l1/bias/Initializer/ConstConst*
valueBd*  ?*-
_class#
!loc:@0/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:d
Е
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
ђ
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias

0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:d
Ј
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b( *
T0
А
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
Р
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
Г
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *
з#<*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Е
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
Є
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

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

:d

70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0
У
0/Actor/target_net/a/a/kernel
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
	container 

$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(
Ј
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
Њ
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*  ?*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
З
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
і
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
_output_shapes
:*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias
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
u
0/Actor/target_net/a/a/ReluRelu0/Actor/target_net/a/a/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
d
0/Actor/target_net/a/scaled_a/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Relu0/Actor/target_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
L
0/mul/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
b
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes
:	d*
T0
N
	0/mul_1/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
d
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
_output_shapes
:	d*
T0
F
0/addAdd0/mul0/mul_1*
T0*
_output_shapes
:	d
Л
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
N
	0/mul_2/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:d
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
:d
E
0/add_1Add0/mul_20/mul_3*
_output_shapes
:d*
T0
Ж

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
 *Єp}?*
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
з#<*
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
Р

0/Assign_2Assign0/Actor/target_net/a/a/kernel0/add_2*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(
N
	0/mul_6/xConst*
valueB
 *Єp}?*
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
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
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
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0

H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2c

70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	

30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes
:	*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Н
0/Critic/eval_net/l1/w1_s
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container 
ћ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
_output_shapes
:	*
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
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 

H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2l*
dtype0*
_output_shapes

:*

seed*
T0
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
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:

0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
Њ
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*  ?**
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
0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
_output_shapes

:*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
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
:*
valueB"      *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
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
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
­
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2~
З
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
 
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Щ
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

'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Б
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
А
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*  ?*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
Н
0/Critic/eval_net/q/dense/bias
VariableV2*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ї
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
Ф
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
М
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
М
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"     *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
Џ
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_s
Б
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
 
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2
Є
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	

50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes
:	*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
С
0/Critic/target_net/l1/w1_s
VariableV2*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	

"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
Ѓ
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	
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
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 

J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2*
dtype0*
_output_shapes

:*

seed
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
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
Ў
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*  ?*,
_class"
 loc:@0/Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
Л
0/Critic/target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/target_net/l1/b1*
	container *
shape
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
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
Ј
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Т
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*'
_output_shapes
:џџџџџџџџџ*
T0

0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
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
 *   ?*5
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
)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
З
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
Д
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*  ?*3
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
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:
Ъ
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Т
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
U
0/target_q/mul/xConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
~
0/target_q/mulMul0/target_q/mul/x#0/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
\
0/target_q/addAddR/r0/target_q/mul*
T0*'
_output_shapes
:џџџџџџџџџ
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
0/C_train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?

0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0

60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Ф
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0

.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
л
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0

00/C_train/gradients/0/TD_error/Mean_grad/Shape_1Shape0/TD_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
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
: *
	keep_dims( *

Tidx0*
T0
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
й
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
Ј
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
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
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
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
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
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
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
:*
	keep_dims( *

Tidx0
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
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1

;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ш
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ы
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
с
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
о
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	

=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
б
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
ш
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ*
T0
х
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1
и
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 

0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Щ
J0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"     *
dtype0
Г
@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
К
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
_output_shapes
:	*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*

index_type0
Ь
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	
 
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Л
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
_output_shapes
:	*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Ы
L0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"     
Е
B0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes
:	
Ю
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	
І
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	
П
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
Н
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    
Ъ
(0/C_train/0/Critic/eval_net/l1/w1_a/Adam
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

/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(
О
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Й
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
Ц
&0/C_train/0/Critic/eval_net/l1/b1/Adam
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
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
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
/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(
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

:*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0
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
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Я
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
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
С
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
г
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
П
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ь
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0
Џ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Х
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
С
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
Ю
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
Е
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Щ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
a
0/C_train/Adam/learning_rateConst*
valueB
 *Зб8*
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
0/C_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Ё
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes
:	
Ђ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:

70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
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
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
Ю
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
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
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
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
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Љ
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
с
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
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
60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0

F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ё
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
N
	0/mul_8/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
e
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
_output_shapes
:	*
T0
N
	0/mul_9/xConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
c
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
_output_shapes
:	*
T0
J
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes
:	
Н

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	
O

0/mul_10/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
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
0/mul_11/x0/Critic/eval_net/l1/w1_a/read*
_output_shapes

:*
T0
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

0/mul_12/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
d
0/mul_12Mul
0/mul_12/x0/Critic/target_net/l1/b1/read*
_output_shapes

:*
T0
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
0/add_6Add0/mul_120/mul_13*
_output_shapes

:*
T0
И

0/Assign_6Assign0/Critic/target_net/l1/b10/add_6*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
O

0/mul_14/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
O

0/mul_15/xConst*
valueB
 *
з#<*
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
Ъ

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

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Relu*
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
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0

=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
В
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Relu0/policy_grads/gradients/Fill*
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
м
@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGradReluGradA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape0/Actor/eval_net/a/a/Relu*
T0*'
_output_shapes
:џџџџџџџџџ
г
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad 0/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0
ї
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0
й
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd*
T0
б
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d

?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
с
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0

#0/A_train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
Ї
0/A_train/beta1_power
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
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
0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 

#0/A_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
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
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 

0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ы
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"  d   *
dtype0*
_output_shapes
:
Е
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
О
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes
:	d
Ю
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	d
Є
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
О
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Э
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"  d   *
dtype0*
_output_shapes
:
З
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	d*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0
а
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	d
Њ
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
Т
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Г
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    
Р
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d

.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Г
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
_output_shapes
:d*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Е
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:d*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    *
dtype0
Т
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 

00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
З
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
С
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
Ю
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
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
Ї
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
Р
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
У
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
а
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d
­
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
Ф
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
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
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    *
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
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
К
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
a
0/A_train/Adam/learning_rateConst*
valueB
 *ЗбИ*
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
0/A_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes
:	d*
use_locking( 

80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( 

;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d

90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
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
0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
Ц
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
К
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
ж
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign"&шк7І     БўS	'Ѕу[uзAJџЬ
Љ
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
shared_namestring *1.14.02v1.14.0-rc1-22-gaf24dc91b5Ъ№
h
S/sPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
f
R/rPlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
j
S_/s_Placeholder*
shape:џџџџџџџџџ*
dtype0*(
_output_shapes
:џџџџџџџџџ
К
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"  d   
­
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
Џ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 

I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	d*

seed*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2
 
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d*
T0

40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
П
0/Actor/eval_net/l1/kernel
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	d
џ
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
 
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
_output_shapes
:	d*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Є
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:d*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*  ?
Б
0/Actor/eval_net/l1/bias
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
ъ
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d

0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Ђ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџd*
transpose_b( *
T0
Њ
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd*
T0
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*'
_output_shapes
:џџџџџџџџџd*
T0
М
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB"d      *
dtype0
Џ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
Б
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 

J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

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

:d

50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
П
0/Actor/eval_net/a/a/kernel
VariableV2*
shared_name *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d

"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
Ђ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
І
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*  ?*
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
q
0/Actor/eval_net/a/a/ReluRelu0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
b
0/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/a/Relu0/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
О
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB"  d   *
dtype0*
_output_shapes
:
Б
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
Г
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *   ?*
dtype0
Ђ
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2(*
dtype0*
_output_shapes
:	d*

seed
Ј
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d*
T0

60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d
У
0/Actor/target_net/l1/kernel
VariableV2*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d

#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
І
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	d
Ј
,0/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@0/Actor/target_net/l1/bias*
valueBd*  ?*
dtype0*
_output_shapes
:d
Е
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
ђ
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(

0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
_output_shapes
:d*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias
Ј
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџd*
transpose_b( *
T0
А
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*'
_output_shapes
:џџџџџџџџџd*
T0
Р
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Г
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *
з#<*
dtype0
Е
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *   ?
Є
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

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

:d

70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
У
0/Actor/target_net/a/a/kernel
VariableV2*
shared_name *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
	container *
shape
:d*
dtype0*
_output_shapes

:d

$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
Ј
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
Њ
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
valueB*  ?*
dtype0*
_output_shapes
:
З
0/Actor/target_net/a/a/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@0/Actor/target_net/a/a/bias*
	container 
і
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(

 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
П
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
Г
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
u
0/Actor/target_net/a/a/ReluRelu0/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/a/Relu0/Actor/target_net/a/scaled_a/y*'
_output_shapes
:џџџџџџџџџ*
T0
L
0/mul/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
b
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes
:	d*
T0
N
	0/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
d
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
_output_shapes
:	d*
T0
F
0/addAdd0/mul0/mul_1*
T0*
_output_shapes
:	d
Л
0/AssignAssign0/Actor/target_net/l1/kernel0/add*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
N
	0/mul_2/xConst*
valueB
 *Єp}?*
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
з#<*
dtype0*
_output_shapes
: 
]
0/mul_3Mul	0/mul_3/x0/Actor/eval_net/l1/bias/read*
_output_shapes
:d*
T0
E
0/add_1Add0/mul_20/mul_3*
T0*
_output_shapes
:d
Ж

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
 *Єp}?*
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
з#<*
dtype0*
_output_shapes
: 
d
0/mul_5Mul	0/mul_5/x 0/Actor/eval_net/a/a/kernel/read*
_output_shapes

:d*
T0
I
0/add_2Add0/mul_40/mul_5*
_output_shapes

:d*
T0
Р

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
 *Єp}?*
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
з#<*
dtype0
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
valueB"     *
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
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *   ?

H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2c

70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	

30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes
:	*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Н
0/Critic/eval_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	
ћ
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(

0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
И
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ћ
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *    
­
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *   ?*
dtype0*
_output_shapes
: 

H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2l
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
0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
Њ
)0/Critic/eval_net/l1/b1/Initializer/ConstConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*  ?*
dtype0*
_output_shapes

:
З
0/Critic/eval_net/l1/b1
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
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Ж
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0

0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ

0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:џџџџџџџџџ
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Й
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *   ?*
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
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0
 
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Щ
 0/Critic/eval_net/q/dense/kernel
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

'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Б
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
А
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*  ?
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
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias
Ф
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
М
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
М
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB"     *
dtype0*
_output_shapes
:
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
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *   ?*
dtype0
 
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2
Є
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	

50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	
С
0/Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape:	

"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(
Ѓ
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
_output_shapes
:	*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
М
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
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
 *   ?*
dtype0*
_output_shapes
: 

J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2*
dtype0*
_output_shapes

:
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
VariableV2*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
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
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
Ў
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*,
_class"
 loc:@0/Critic/target_net/l1/b1*
valueB*  ?*
dtype0
Л
0/Critic/target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/target_net/l1/b1*
	container *
shape
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
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
_output_shapes

:
Ј
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
T0
Т
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
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
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Д
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2Ј*
dtype0*
_output_shapes

:
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
dtype0*
_output_shapes

:*
shared_name *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:

)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
З
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Д
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
valueB*  ?
С
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
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ
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
0/TD_error/ConstConst*
_output_shapes
:*
valueB"       *
dtype0

0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
0/C_train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
b
0/C_train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?

0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 

60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
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
00/C_train/gradients/0/TD_error/Mean_grad/Shape_1Shape0/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
й
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
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
out_type0*
_output_shapes
:*
T0
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
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
т
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
и
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
ъ
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
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
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
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
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ

B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
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
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
ш
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
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
70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
њ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ї
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0
Ъ
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
о
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
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
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
є
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1

;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *(
_output_shapes
:џџџџџџџџџ*
transpose_b(
ш
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	*
transpose_b( *
T0
Ы
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
с
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul
о
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1

=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
§
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
transpose_a(*
_output_shapes

:*
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
VariableV2*
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: 
и
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1

0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1

#0/C_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: **
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *wО?
Ѕ
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
и
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1

0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
Щ
J0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Г
@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
К
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
Ь
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
 
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Л
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
Ы
L0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Е
B0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Р
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
Ю
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	
І
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	
П
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	
Н
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
Ъ
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
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:*
T0
П
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0
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
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ц
&0/C_train/0/Critic/eval_net/l1/b1/Adam
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
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ш
(0/C_train/0/Critic/eval_net/l1/b1/Adam_1
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
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Э
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0
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
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
Х
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
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
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
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
 *Зб8*
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
0/C_train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Ё
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Ђ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:

70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
Ш
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( 
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
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
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
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
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
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Љ
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
с
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b(
Ю
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
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
valueB"      *
dtype0*
_output_shapes
:

H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ї
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0

F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
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
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
	0/mul_8/xConst*
valueB
 *Єp}?*
dtype0*
_output_shapes
: 
e
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes
:	
N
	0/mul_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *
з#<
c
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes
:	
J
0/add_4Add0/mul_80/mul_9*
_output_shapes
:	*
T0
Н

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	
O

0/mul_10/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
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
0/add_5Add0/mul_100/mul_11*
_output_shapes

:*
T0
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

0/mul_12/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
d
0/mul_12Mul
0/mul_12/x0/Critic/target_net/l1/b1/read*
_output_shapes

:*
T0
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

0/Assign_6Assign0/Critic/target_net/l1/b10/add_6*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
O

0/mul_14/xConst*
_output_shapes
: *
valueB
 *Єp}?*
dtype0
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
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
0/add_7Add0/mul_140/mul_15*
T0*
_output_shapes

:
Ъ

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
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ*
T0*

index_type0

?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/a/Relu*
T0*
out_type0*
_output_shapes
:

A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Љ
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
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
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
В
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/a/Relu0/policy_grads/gradients/Fill*'
_output_shapes
:џџџџџџџџџ*
T0
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
м
@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGradReluGradA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape0/Actor/eval_net/a/a/Relu*
T0*'
_output_shapes
:џџџџџџџџџ
г
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0

@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad 0/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџd*
transpose_b(
ї
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
transpose_a(*
_output_shapes

:d*
transpose_b( *
T0
й
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџd
б
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:d*
T0

?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:џџџџџџџџџ*
transpose_b(
с
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes
:	d*
transpose_b( 

#0/A_train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *fff?
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
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 

#0/A_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *wО?
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
0/A_train/beta2_power/readIdentity0/A_train/beta2_power*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
Ы
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
Е
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
О
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Ю
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
Є
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
О
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Э
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
З
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Ф
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
а
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	d
Њ
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	d
Т
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	d
Г
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
Р
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

.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Г
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Е
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
Т
)0/A_train/0/Actor/eval_net/l1/bias/Adam_1
VariableV2*
shared_name *+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d

00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
З
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
С
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
_output_shapes

:d*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0
Ю
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
Ї
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
Р
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0
У
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
а
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
­
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
Ф
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
Е
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
Т
(0/A_train/0/Actor/eval_net/a/a/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:
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
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0
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
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
 *ЗбИ*
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
0/A_train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wЬ+2

:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_nesterov( *
_output_shapes
:	d*
use_locking( *
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel

80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d*
use_locking( 

;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d*
use_locking( *
T0

90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:

0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Т
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(

0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
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
ж
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign"&"А
trainable_variables
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
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08".
train_op"
 
0/C_train/Adam
0/A_train/Adam"9
	variables99
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
,0/A_train/0/Actor/eval_net/a/a/bias/Adam_1:010/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/read:02>0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:05аD