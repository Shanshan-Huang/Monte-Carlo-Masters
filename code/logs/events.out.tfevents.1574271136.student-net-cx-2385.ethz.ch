       £K"	   (^u„Abrain.Event:2?≤Зl
Е     ЕX$	…t4(^u„A"эЙ
h
S/sPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С
f
R/rPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
j
S_/s_Placeholder*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С*
dtype0
Ї
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"С  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
≠
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *Ј—8*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
ѓ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЌћL=*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Ь
I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes
:	Сd*

seed*
T0
†
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
Й
40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
њ
0/Actor/eval_net/l1/kernel
VariableV2*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd
€
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
†
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
§
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueBd*  А?*+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
±
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
к
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Х
0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Ґ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
™
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
Љ
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
ѓ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *Ј—8*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
±
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *ЌћL=*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
Ю
J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
seed2
£
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
М
50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
њ
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
В
"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
Ґ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
¶
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*  А?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
≥
0/Actor/eval_net/a/a/bias
VariableV2*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape:*
dtype0*
_output_shapes
:
о
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ш
0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
є
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
≠
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
q
0/Actor/eval_net/a/a/ReluRelu0/Actor/eval_net/a/a/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
a
0/Actor/eval_net/a/truediv/yConst*
valueB
 *  »B*
dtype0*
_output_shapes
: 
Р
0/Actor/eval_net/a/truedivRealDiv0/Actor/eval_net/a/a/Relu0/Actor/eval_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
b
0/Actor/eval_net/a/scaled_a/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
П
0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/truediv0/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:€€€€€€€€€*
T0
Њ
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"С  d   */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
±
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *Ј—8*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
≥
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЌћL=*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
Ґ
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*

seed*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2**
dtype0*
_output_shapes
:	Сd
®
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
С
60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
√
0/Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape:	Сd
З
#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
¶
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
®
,0/Actor/target_net/l1/bias/Initializer/ConstConst*
valueBd*  А?*-
_class#
!loc:@0/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:d
µ
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
т
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ы
0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:d
®
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( *
T0
∞
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*'
_output_shapes
:€€€€€€€€€d*
T0*
data_formatNHWC
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
ј
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
:
≥
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *Ј—8*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
µ
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЌћL=*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
§
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed2:*
dtype0*
_output_shapes

:d*

seed*
T0
Ђ
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
Ф
70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
√
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
К
$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
®
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
™
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*  А?*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
Ј
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
ц
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ю
 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:
њ
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
≥
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
u
0/Actor/target_net/a/a/ReluRelu0/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
c
0/Actor/target_net/a/truediv/yConst*
valueB
 *  »B*
dtype0*
_output_shapes
: 
Ц
0/Actor/target_net/a/truedivRealDiv0/Actor/target_net/a/a/Relu0/Actor/target_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/truediv0/Actor/target_net/a/scaled_a/y*'
_output_shapes
:€€€€€€€€€*
T0
L
0/mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
b
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
T0*
_output_shapes
:	Сd
N
	0/mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
d
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes
:	Сd
F
0/addAdd0/mul0/mul_1*
T0*
_output_shapes
:	Сd
ї
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
N
	0/mul_2/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
_
0/mul_2Mul	0/mul_2/x0/Actor/target_net/l1/bias/read*
_output_shapes
:d*
T0
N
	0/mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
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
ґ

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
N
	0/mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
f
0/mul_4Mul	0/mul_4/x"0/Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:d
N
	0/mul_5/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
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
ј

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
 *§p}?*
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
„#<*
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
Є

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
:€€€€€€€€€
Є
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"С     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0
Ђ
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
≠
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Щ
H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	С*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2g
Ь
70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Е
30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
љ
0/Critic/eval_net/l1/w1_s
VariableV2*
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container 
ы
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
validate_shape(*
_output_shapes
:	С*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Э
0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Є
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ђ
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
: 
≠
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Ш
H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2p
Ы
70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Д
30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
ї
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
ъ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ь
0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
™
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
valueB*  А?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
Ј
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
к
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ц
0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ґ
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
ґ
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Н
0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:€€€€€€€€€
Л
0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€*
T0
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*'
_output_shapes
:€€€€€€€€€*
T0
∆
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
:
є
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
ї
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
Ѓ
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2В*
dtype0*
_output_shapes

:*

seed
Ј
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
†
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
…
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
Ц
'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
±
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
∞
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*  А?*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
љ
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
В
%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
І
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
ƒ
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Љ
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Љ
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"С     *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
ѓ
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
±
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
†
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
_output_shapes
:	С*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2С*
dtype0
§
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes
:	С*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
Н
50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Ѕ
0/Critic/target_net/l1/w1_s
VariableV2*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С
Г
"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
£
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Љ
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
ѓ
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@0/Critic/target_net/l1/w1_a
±
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Я
J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2Ъ
£
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
М
50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
њ
0/Critic/target_net/l1/w1_a
VariableV2*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
	container *
shape
:*
dtype0*
_output_shapes

:
В
"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ґ
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:
Ѓ
+0/Critic/target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:*
valueB*  А?*,
_class"
 loc:@0/Critic/target_net/l1/b1
ї
0/Critic/target_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@0/Critic/target_net/l1/b1
т
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
Ь
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
_output_shapes

:
®
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
¬
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
У
0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
С
0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
 
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
љ
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0
њ
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
dtype0
і
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
seed2ђ*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
њ
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
®
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ќ
"0/Critic/target_net/q/dense/kernel
VariableV2*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ю
)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(
Ј
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
і
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*  А?*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
Ѕ
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
К
'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(
≠
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias
 
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
¬
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
\
0/target_q/addAddR/r0/target_q/mul*'
_output_shapes
:€€€€€€€€€*
T0
Ц
0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
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
 *  А?*
dtype0*
_output_shapes
: 
Н
0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
З
60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
ƒ
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
К
.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
џ
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
М
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
.0/C_train/gradients/0/TD_error/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
’
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
ў
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
20/C_train/gradients/0/TD_error/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ѕ
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
њ
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
_output_shapes
: *
T0
®
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
Ћ
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Й
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
T0*
out_type0*
_output_shapes
:
Ю
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
в
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
Ў
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
к
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:€€€€€€€€€*
T0
К
90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
А
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
О
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ж
?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
≥
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
 
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
ж
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
а
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg
г
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
п
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
€
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
ч
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ю
@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
Л
B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Џ
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
ф
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
с
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
и
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
С
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
М
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ч
I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ж
70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ъ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
К
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ч
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
 
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
ё
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
џ
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
Т
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Ц
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
С
G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
У
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Ч
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ƒ
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
÷
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape
№
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
К
;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*(
_output_shapes
:€€€€€€€€€С*
transpose_a( *
transpose_b(*
T0
и
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	С*
transpose_a(*
transpose_b( 
Ћ
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
б
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€С
ё
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	С
Н
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
э
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
—
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
и
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul
е
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:*
T0
Ф
#0/C_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
•
0/C_train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1
Ў
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
К
0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Ф
#0/C_train/beta2_power/initial_valueConst*
valueB
 *wЊ?**
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
•
0/C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: 
Ў
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
К
0/C_train/beta2_power/readIdentity0/C_train/beta2_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
…
J0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
≥
@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes
:	С*
T0
ћ
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С
†
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
ї
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Ћ
L0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
µ
B0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes
:	С
ќ
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	С
¶
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
њ
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
_output_shapes
:	С*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
љ
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
 
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
Я
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(
Ї
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
њ
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
ћ
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
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
•
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Њ
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
є
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
∆
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
Ч
-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
і
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
ї
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
»
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
Э
/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Є
-0/C_train/0/Critic/eval_net/l1/b1/Adam_1/readIdentity(0/C_train/0/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ћ
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0
Ў
/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
ї
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
ѕ
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Ќ
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
Џ
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
Ѕ
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(
”
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
њ
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
ћ
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ѓ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
≈
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ѕ
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
ќ
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
µ
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
…
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
a
0/C_train/Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Y
0/C_train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Y
0/C_train/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
0/C_train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
°
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
use_nesterov( *
_output_shapes
:	С
Ґ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Х
70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( 
»
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
ї
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
ќ
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
ј
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
–
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
ƒ
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
ю
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
 *  А?*
dtype0*
_output_shapes
: 
Ы
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
©
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
б
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(*
T0
ќ
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
‘
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€*
T0
Р
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
Л
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ф
H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Г
60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
З
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
С
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Х
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0
О
F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
с
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Г
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ъ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
к
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
N
	0/mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
e
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
_output_shapes
:	С*
T0
N
	0/mul_9/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
c
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
_output_shapes
:	С*
T0
J
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes
:	С
љ

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
validate_shape(*
_output_shapes
:	С*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s
O

0/mul_10/xConst*
valueB
 *§p}?*
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
„#<*
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
Љ

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
 *§p}?*
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
„#<*
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
Є

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
 *§p}?*
dtype0
m
0/mul_14Mul
0/mul_14/x'0/Critic/target_net/q/dense/kernel/read*
T0*
_output_shapes

:
O

0/mul_15/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
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
 

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
O

0/mul_16/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
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
„#<*
dtype0*
_output_shapes
: 
e
0/mul_17Mul
0/mul_17/x#0/Critic/eval_net/q/dense/bias/read*
_output_shapes
:*
T0
G
0/add_8Add0/mul_160/mul_17*
_output_shapes
:*
T0
¬

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
g
"0/policy_grads/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
≠
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
Щ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/truediv*
T0*
out_type0*
_output_shapes
:
Д
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
©
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Ф
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
≥
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/truediv0/policy_grads/gradients/Fill*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ч
>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/ShapeShape0/Actor/eval_net/a/a/Relu*
_output_shapes
:*
T0*
out_type0
Г
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
¶
N0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ё
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDivRealDivA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape0/Actor/eval_net/a/truediv/y*'
_output_shapes
:€€€€€€€€€*
T0
Х
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/SumSum@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDivN0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/ReshapeReshape<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Р
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/NegNeg0/Actor/eval_net/a/a/Relu*
T0*'
_output_shapes
:€€€€€€€€€
џ
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_1RealDiv<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Neg0/Actor/eval_net/a/truediv/y*'
_output_shapes
:€€€€€€€€€*
T0
б
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_2RealDivB0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_10/Actor/eval_net/a/truediv/y*'
_output_shapes
:€€€€€€€€€*
T0
ь
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/mulMulA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeB0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:€€€€€€€€€
Х
>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum_1Sum<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/mulP0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ю
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Reshape_1Reshape>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum_1@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
џ
@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGradReluGrad@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Reshape0/Actor/eval_net/a/a/Relu*'
_output_shapes
:€€€€€€€€€*
T0
”
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ж
@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad 0/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(
ч
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0
ў
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€d
—
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Д
?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€С*
transpose_a( *
transpose_b(
б
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
_output_shapes
:	Сd*
transpose_a(*
transpose_b( 
Ц
#0/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
І
0/A_train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: 
Џ
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
М
0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
#0/A_train/beta2_power/initial_valueConst*
valueB
 *wЊ?*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
І
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
Џ
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
М
0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ћ
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"С  d   
µ
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes
:	Сd*
T0
ќ
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	Сd
§
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
Њ
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0
Ќ
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
Ј
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ƒ
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes
:	Сd
–
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd*
shared_name 
™
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd*
use_locking(
¬
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0
≥
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    *
dtype0*
_output_shapes
:d
ј
'0/A_train/0/Actor/eval_net/l1/bias/Adam
VariableV2*+
_class!
loc:@0/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
Ч
.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
≥
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
µ
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*    *
dtype0*
_output_shapes
:d
¬
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
Э
00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ј
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Ѕ
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
ќ
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
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
І
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
ј
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
√
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
–
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
≠
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
ƒ
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
µ
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
¬
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
Ы
/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ґ
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
_output_shapes
:*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
Ј
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
ƒ
*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
°
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ї
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
a
0/A_train/Adam/learning_rateConst*
valueB
 *oЇ*
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
0/A_train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЊ?
[
0/A_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Ш
:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes
:	Сd*
use_locking( 
Н
80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d
Э
;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d*
use_locking( 
У
90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
М
0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
¬
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
О
0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
∆
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
Ї
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
÷
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign"&ес§h?≥     Іg2	76(^u„AJ≤ж
©В
:
Add
x"T
y"T
z"T"
Ttype:
2	
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
Н
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
2	Р
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
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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

2	Р
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.14.02v1.14.0-rc1-22-gaf24dc91b5эЙ
h
S/sPlaceholder*
shape:€€€€€€€€€С*
dtype0*(
_output_shapes
:€€€€€€€€€С
f
R/rPlaceholder*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€*
dtype0
j
S_/s_Placeholder*
dtype0*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С
Ї
:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
≠
90/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
ѓ
;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
valueB
 *ЌћL=
Ь
I0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:0/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes
:	Сd*

seed*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
†
80/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI0/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;0/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
Й
40/Actor/eval_net/l1/kernel/Initializer/random_normalAdd80/Actor/eval_net/l1/kernel/Initializer/random_normal/mul90/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
њ
0/Actor/eval_net/l1/kernel
VariableV2*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel
€
!0/Actor/eval_net/l1/kernel/AssignAssign0/Actor/eval_net/l1/kernel40/Actor/eval_net/l1/kernel/Initializer/random_normal*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd*
use_locking(
†
0/Actor/eval_net/l1/kernel/readIdentity0/Actor/eval_net/l1/kernel*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0
§
*0/Actor/eval_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@0/Actor/eval_net/l1/bias*
valueBd*  А?*
dtype0*
_output_shapes
:d
±
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
к
0/Actor/eval_net/l1/bias/AssignAssign0/Actor/eval_net/l1/bias*0/Actor/eval_net/l1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias
Х
0/Actor/eval_net/l1/bias/readIdentity0/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Ґ
0/Actor/eval_net/l1/MatMulMatMulS/s0/Actor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b( *
T0
™
0/Actor/eval_net/l1/BiasAddBiasAdd0/Actor/eval_net/l1/MatMul0/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
o
0/Actor/eval_net/l1/TanhTanh0/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
Љ
;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
ѓ
:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
±
<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
valueB
 *ЌћL=*
dtype0
Ю
J0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
£
90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ0/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<0/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
М
50/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd90/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:0/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
њ
0/Actor/eval_net/a/a/kernel
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
В
"0/Actor/eval_net/a/a/kernel/AssignAssign0/Actor/eval_net/a/a/kernel50/Actor/eval_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
Ґ
 0/Actor/eval_net/a/a/kernel/readIdentity0/Actor/eval_net/a/a/kernel*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
¶
+0/Actor/eval_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB*  А?*
dtype0*
_output_shapes
:
≥
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
о
 0/Actor/eval_net/a/a/bias/AssignAssign0/Actor/eval_net/a/a/bias+0/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ш
0/Actor/eval_net/a/a/bias/readIdentity0/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
є
0/Actor/eval_net/a/a/MatMulMatMul0/Actor/eval_net/l1/Tanh 0/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
≠
0/Actor/eval_net/a/a/BiasAddBiasAdd0/Actor/eval_net/a/a/MatMul0/Actor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
q
0/Actor/eval_net/a/a/ReluRelu0/Actor/eval_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
a
0/Actor/eval_net/a/truediv/yConst*
_output_shapes
: *
valueB
 *  »B*
dtype0
Р
0/Actor/eval_net/a/truedivRealDiv0/Actor/eval_net/a/a/Relu0/Actor/eval_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
b
0/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
П
0/Actor/eval_net/a/scaled_aMul0/Actor/eval_net/a/truediv0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Њ
<0/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
±
;0/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
≥
=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
Ґ
K0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<0/Actor/target_net/l1/kernel/Initializer/random_normal/shape*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
seed2**
dtype0*
_output_shapes
:	Сd*

seed*
T0
®
:0/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK0/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=0/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes
:	Сd*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
С
60/Actor/target_net/l1/kernel/Initializer/random_normalAdd:0/Actor/target_net/l1/kernel/Initializer/random_normal/mul;0/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes
:	Сd*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel
√
0/Actor/target_net/l1/kernel
VariableV2*
shared_name */
_class%
#!loc:@0/Actor/target_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd
З
#0/Actor/target_net/l1/kernel/AssignAssign0/Actor/target_net/l1/kernel60/Actor/target_net/l1/kernel/Initializer/random_normal*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd*
use_locking(*
T0
¶
!0/Actor/target_net/l1/kernel/readIdentity0/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
®
,0/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@0/Actor/target_net/l1/bias*
valueBd*  А?*
dtype0*
_output_shapes
:d
µ
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
т
!0/Actor/target_net/l1/bias/AssignAssign0/Actor/target_net/l1/bias,0/Actor/target_net/l1/bias/Initializer/Const*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
Ы
0/Actor/target_net/l1/bias/readIdentity0/Actor/target_net/l1/bias*-
_class#
!loc:@0/Actor/target_net/l1/bias*
_output_shapes
:d*
T0
®
0/Actor/target_net/l1/MatMulMatMulS_/s_!0/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b( *
T0
∞
0/Actor/target_net/l1/BiasAddBiasAdd0/Actor/target_net/l1/MatMul0/Actor/target_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
s
0/Actor/target_net/l1/TanhTanh0/Actor/target_net/l1/BiasAdd*'
_output_shapes
:€€€€€€€€€d*
T0
ј
=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
≥
<0/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *Ј—8
µ
>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
§
L0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=0/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
seed2:*
dtype0*
_output_shapes

:d*

seed
Ђ
;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL0/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>0/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
Ф
70/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;0/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<0/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel
√
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
К
$0/Actor/target_net/a/a/kernel/AssignAssign0/Actor/target_net/a/a/kernel70/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
®
"0/Actor/target_net/a/a/kernel/readIdentity0/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@0/Actor/target_net/a/a/kernel*
_output_shapes

:d
™
-0/Actor/target_net/a/a/bias/Initializer/ConstConst*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
valueB*  А?*
dtype0*
_output_shapes
:
Ј
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
ц
"0/Actor/target_net/a/a/bias/AssignAssign0/Actor/target_net/a/a/bias-0/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ю
 0/Actor/target_net/a/a/bias/readIdentity0/Actor/target_net/a/a/bias*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
_output_shapes
:*
T0
њ
0/Actor/target_net/a/a/MatMulMatMul0/Actor/target_net/l1/Tanh"0/Actor/target_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
≥
0/Actor/target_net/a/a/BiasAddBiasAdd0/Actor/target_net/a/a/MatMul 0/Actor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
u
0/Actor/target_net/a/a/ReluRelu0/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
c
0/Actor/target_net/a/truediv/yConst*
valueB
 *  »B*
dtype0*
_output_shapes
: 
Ц
0/Actor/target_net/a/truedivRealDiv0/Actor/target_net/a/a/Relu0/Actor/target_net/a/truediv/y*'
_output_shapes
:€€€€€€€€€*
T0
d
0/Actor/target_net/a/scaled_a/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
0/Actor/target_net/a/scaled_aMul0/Actor/target_net/a/truediv0/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
L
0/mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
b
0/mulMul0/mul/x!0/Actor/target_net/l1/kernel/read*
_output_shapes
:	Сd*
T0
N
	0/mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
d
0/mul_1Mul	0/mul_1/x0/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes
:	Сd
F
0/addAdd0/mul0/mul_1*
_output_shapes
:	Сd*
T0
ї
0/AssignAssign0/Actor/target_net/l1/kernel0/add*
use_locking(*
T0*/
_class%
#!loc:@0/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
N
	0/mul_2/xConst*
valueB
 *§p}?*
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
„#<*
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
ґ

0/Assign_1Assign0/Actor/target_net/l1/bias0/add_1*
T0*-
_class#
!loc:@0/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
N
	0/mul_4/xConst*
valueB
 *§p}?*
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
„#<*
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
ј

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
 *§p}?*
dtype0*
_output_shapes
: 
`
0/mul_6Mul	0/mul_6/x 0/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	0/mul_7/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
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
Є

0/Assign_3Assign0/Actor/target_net/a/a/bias0/add_3*
T0*.
_class$
" loc:@0/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
0/Critic/StopGradientStopGradient0/Actor/eval_net/a/scaled_a*
T0*'
_output_shapes
:€€€€€€€€€
Є
90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
Ђ
80/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Щ
H0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
seed2g*
dtype0*
_output_shapes
:	С
Ь
70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С*
T0
Е
30/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes
:	С*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
љ
0/Critic/eval_net/l1/w1_s
VariableV2*
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container 
ы
 0/Critic/eval_net/l1/w1_s/AssignAssign0/Critic/eval_net/l1/w1_s30/Critic/eval_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
Э
0/Critic/eval_net/l1/w1_s/readIdentity0/Critic/eval_net/l1/w1_s*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С*
T0
Є
90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ђ
80/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
_output_shapes
: *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
valueB
 *   ?*
dtype0
Ш
H0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal90/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
seed2p
Ы
70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH0/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:0/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
Д
30/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd70/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul80/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
ї
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
ъ
 0/Critic/eval_net/l1/w1_a/AssignAssign0/Critic/eval_net/l1/w1_a30/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ь
0/Critic/eval_net/l1/w1_a/readIdentity0/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
™
)0/Critic/eval_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB*  А?
Ј
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
к
0/Critic/eval_net/l1/b1/AssignAssign0/Critic/eval_net/l1/b1)0/Critic/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Ц
0/Critic/eval_net/l1/b1/readIdentity0/Critic/eval_net/l1/b1*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
Ґ
0/Critic/eval_net/l1/MatMulMatMulS/s0/Critic/eval_net/l1/w1_s/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
ґ
0/Critic/eval_net/l1/MatMul_1MatMul0/Critic/StopGradient0/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
Н
0/Critic/eval_net/l1/addAdd0/Critic/eval_net/l1/MatMul0/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
Л
0/Critic/eval_net/l1/add_1Add0/Critic/eval_net/l1/add0/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
o
0/Critic/eval_net/l1/ReluRelu0/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
∆
@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
є
?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ї
A0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ѓ
O0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*

seed*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
seed2В*
dtype0*
_output_shapes

:
Ј
>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
†
:0/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?0/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
…
 0/Critic/eval_net/q/dense/kernel
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Ц
'0/Critic/eval_net/q/dense/kernel/AssignAssign 0/Critic/eval_net/q/dense/kernel:0/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
±
%0/Critic/eval_net/q/dense/kernel/readIdentity 0/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
∞
00/Critic/eval_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
valueB*  А?*
dtype0*
_output_shapes
:
љ
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
В
%0/Critic/eval_net/q/dense/bias/AssignAssign0/Critic/eval_net/q/dense/bias00/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
І
#0/Critic/eval_net/q/dense/bias/readIdentity0/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
ƒ
 0/Critic/eval_net/q/dense/MatMulMatMul0/Critic/eval_net/l1/Relu%0/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
Љ
!0/Critic/eval_net/q/dense/BiasAddBiasAdd 0/Critic/eval_net/q/dense/MatMul#0/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Љ
;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB"С     
ѓ
:0/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
±
<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
valueB
 *   ?*
dtype0*
_output_shapes
: 
†
J0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	С*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
seed2С
§
90/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Н
50/Critic/target_net/l1/w1_s/Initializer/random_normalAdd90/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Ѕ
0/Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes
:	С*
shared_name *.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
	container *
shape:	С
Г
"0/Critic/target_net/l1/w1_s/AssignAssign0/Critic/target_net/l1/w1_s50/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
£
 0/Critic/target_net/l1/w1_s/readIdentity0/Critic/target_net/l1/w1_s*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Љ
;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB"      *
dtype0
ѓ
:0/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *    
±
<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Я
J0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;0/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*

seed*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
seed2Ъ*
dtype0*
_output_shapes

:
£
90/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ0/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<0/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a
М
50/Critic/target_net/l1/w1_a/Initializer/random_normalAdd90/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:0/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
њ
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
В
"0/Critic/target_net/l1/w1_a/AssignAssign0/Critic/target_net/l1/w1_a50/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ґ
 0/Critic/target_net/l1/w1_a/readIdentity0/Critic/target_net/l1/w1_a*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
Ѓ
+0/Critic/target_net/l1/b1/Initializer/ConstConst*,
_class"
 loc:@0/Critic/target_net/l1/b1*
valueB*  А?*
dtype0*
_output_shapes

:
ї
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
т
 0/Critic/target_net/l1/b1/AssignAssign0/Critic/target_net/l1/b1+0/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
Ь
0/Critic/target_net/l1/b1/readIdentity0/Critic/target_net/l1/b1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/target_net/l1/b1
®
0/Critic/target_net/l1/MatMulMatMulS_/s_ 0/Critic/target_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
¬
0/Critic/target_net/l1/MatMul_1MatMul0/Actor/target_net/a/scaled_a 0/Critic/target_net/l1/w1_a/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
У
0/Critic/target_net/l1/addAdd0/Critic/target_net/l1/MatMul0/Critic/target_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
С
0/Critic/target_net/l1/add_1Add0/Critic/target_net/l1/add0/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
s
0/Critic/target_net/l1/ReluRelu0/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
 
B0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
љ
A0/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0
њ
C0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
і
Q0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB0/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
seed2ђ
њ
@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ0/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC0/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
®
<0/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA0/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ќ
"0/Critic/target_net/q/dense/kernel
VariableV2*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
Ю
)0/Critic/target_net/q/dense/kernel/AssignAssign"0/Critic/target_net/q/dense/kernel<0/Critic/target_net/q/dense/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
Ј
'0/Critic/target_net/q/dense/kernel/readIdentity"0/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
і
20/Critic/target_net/q/dense/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
valueB*  А?
Ѕ
 0/Critic/target_net/q/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
	container 
К
'0/Critic/target_net/q/dense/bias/AssignAssign 0/Critic/target_net/q/dense/bias20/Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
≠
%0/Critic/target_net/q/dense/bias/readIdentity 0/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
_output_shapes
:
 
"0/Critic/target_net/q/dense/MatMulMatMul0/Critic/target_net/l1/Relu'0/Critic/target_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
¬
#0/Critic/target_net/q/dense/BiasAddBiasAdd"0/Critic/target_net/q/dense/MatMul%0/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€*
T0
\
0/target_q/addAddR/r0/target_q/mul*
T0*'
_output_shapes
:€€€€€€€€€
Ц
0/TD_error/SquaredDifferenceSquaredDifference0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
a
0/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
0/TD_error/MeanMean0/TD_error/SquaredDifference0/TD_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
0/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
0/C_train/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Н
0/C_train/gradients/FillFill0/C_train/gradients/Shape0/C_train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
З
60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ƒ
00/C_train/gradients/0/TD_error/Mean_grad/ReshapeReshape0/C_train/gradients/Fill60/C_train/gradients/0/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
К
.0/C_train/gradients/0/TD_error/Mean_grad/ShapeShape0/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
џ
-0/C_train/gradients/0/TD_error/Mean_grad/TileTile00/C_train/gradients/0/TD_error/Mean_grad/Reshape.0/C_train/gradients/0/TD_error/Mean_grad/Shape*'
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
М
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
’
-0/C_train/gradients/0/TD_error/Mean_grad/ProdProd00/C_train/gradients/0/TD_error/Mean_grad/Shape_1.0/C_train/gradients/0/TD_error/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
z
00/C_train/gradients/0/TD_error/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
ў
/0/C_train/gradients/0/TD_error/Mean_grad/Prod_1Prod00/C_train/gradients/0/TD_error/Mean_grad/Shape_200/C_train/gradients/0/TD_error/Mean_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
20/C_train/gradients/0/TD_error/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ѕ
00/C_train/gradients/0/TD_error/Mean_grad/MaximumMaximum/0/C_train/gradients/0/TD_error/Mean_grad/Prod_120/C_train/gradients/0/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
њ
10/C_train/gradients/0/TD_error/Mean_grad/floordivFloorDiv-0/C_train/gradients/0/TD_error/Mean_grad/Prod00/C_train/gradients/0/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
®
-0/C_train/gradients/0/TD_error/Mean_grad/CastCast10/C_train/gradients/0/TD_error/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 
Ћ
00/C_train/gradients/0/TD_error/Mean_grad/truedivRealDiv-0/C_train/gradients/0/TD_error/Mean_grad/Tile-0/C_train/gradients/0/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Й
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/ShapeShape0/target_q/add*
T0*
out_type0*
_output_shapes
:
Ю
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1Shape!0/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalarConst1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
в
90/C_train/gradients/0/TD_error/SquaredDifference_grad/MulMul<0/C_train/gradients/0/TD_error/SquaredDifference_grad/scalar00/C_train/gradients/0/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
Ў
90/C_train/gradients/0/TD_error/SquaredDifference_grad/subSub0/target_q/add!0/Critic/eval_net/q/dense/BiasAdd1^0/C_train/gradients/0/TD_error/Mean_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
к
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/Mul90/C_train/gradients/0/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:€€€€€€€€€*
T0
К
90/C_train/gradients/0/TD_error/SquaredDifference_grad/SumSum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1K0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
А
=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeReshape90/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
О
;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1Sum;0/C_train/gradients/0/TD_error/SquaredDifference_grad/mul_1M0/C_train/gradients/0/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ж
?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1Reshape;0/C_train/gradients/0/TD_error/SquaredDifference_grad/Sum_1=0/C_train/gradients/0/TD_error/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
≥
90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegNeg?0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
 
F0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg>^0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape
ж
N0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/TD_error/SquaredDifference_grad/ReshapeG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
а
P0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity90/C_train/gradients/0/TD_error/SquaredDifference_grad/NegG^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
г
F0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
п
K0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1
€
S0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP0/C_train/gradients/0/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
ч
U0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*Y
_classO
MKloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
Ю
@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%0/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
Л
B0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/ReluS0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
Џ
J0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulC^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
ф
R0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulK^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*S
_classI
GEloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul
с
T0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
и
;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency0/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
С
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
М
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ч
I0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ж
70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradI0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ъ
;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape70/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
К
90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum;0/C_train/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradK0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ч
=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape90/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
 
D0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape>^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
ё
L0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeE^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
џ
N0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1E^0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1
Т
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Ц
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
С
G0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape90/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
У
50/C_train/gradients/0/Critic/eval_net/l1/add_grad/SumSumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape50/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Ч
70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_1SumL0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI0/C_train/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape70/C_train/gradients/0/Critic/eval_net/l1/add_grad/Sum_190/C_train/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ƒ
B0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape<^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1
÷
J0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity90/C_train/gradients/0/Critic/eval_net/l1/add_grad/ReshapeC^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€
№
L0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1C^0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
К
;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency0/Critic/eval_net/l1/w1_s/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€С
и
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	С
Ћ
E0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1
б
M0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMulF^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€С
ё
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1F^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	С
Н
=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_10/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(*
T0
э
?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradientL0/C_train/gradients/0/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
transpose_a(*
_output_shapes

:*
transpose_b( *
T0
—
G0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul@^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
и
O0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulH^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:€€€€€€€€€
е
Q0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
Ф
#0/C_train/beta1_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
•
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
Ў
0/C_train/beta1_power/AssignAssign0/C_train/beta1_power#0/C_train/beta1_power/initial_value**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
К
0/C_train/beta1_power/readIdentity0/C_train/beta1_power*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: 
Ф
#0/C_train/beta2_power/initial_valueConst**
_class 
loc:@0/Critic/eval_net/l1/b1*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
•
0/C_train/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@0/Critic/eval_net/l1/b1*
	container *
shape: 
Ў
0/C_train/beta2_power/AssignAssign0/C_train/beta2_power#0/C_train/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
К
0/C_train/beta2_power/readIdentity0/C_train/beta2_power**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
…
J0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"С     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
≥
@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Ї
:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*

index_type0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С*
T0
ћ
(0/C_train/0/Critic/eval_net/l1/w1_s/Adam
VariableV2*
shared_name *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С
†
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
ї
-0/C_train/0/Critic/eval_net/l1/w1_s/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*
_output_shapes
:	С*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Ћ
L0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"С     *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
µ
B0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
ј
<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
_output_shapes
:	С*
T0*

index_type0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
ќ
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name 
¶
10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
_output_shapes
:	С*
use_locking(*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
validate_shape(
њ
/0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
љ
:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0
 
(0/C_train/0/Critic/eval_net/l1/w1_a/Adam
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
Я
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/AssignAssign(0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ї
-0/C_train/0/Critic/eval_net/l1/w1_a/Adam/readIdentity(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
_output_shapes

:
њ
<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
ћ
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1
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
•
10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1<0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Њ
/0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a
є
80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
∆
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
Ч
-0/C_train/0/Critic/eval_net/l1/b1/Adam/AssignAssign&0/C_train/0/Critic/eval_net/l1/b1/Adam80/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(
і
+0/C_train/0/Critic/eval_net/l1/b1/Adam/readIdentity&0/C_train/0/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes

:
ї
:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
valueB*    **
_class 
loc:@0/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
»
(0/C_train/0/Critic/eval_net/l1/b1/Adam_1
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
Э
/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/AssignAssign(0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Є
-0/C_train/0/Critic/eval_net/l1/b1/Adam_1/readIdentity(0/C_train/0/Critic/eval_net/l1/b1/Adam_1*
_output_shapes

:*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
Ћ
A0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Ў
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
ї
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/kernel/AdamA0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
ѕ
40/C_train/0/Critic/eval_net/q/dense/kernel/Adam/readIdentity/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Ќ
C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
Џ
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
Ѕ
80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
”
60/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1*
_output_shapes

:*
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel
њ
?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
ћ
-0/C_train/0/Critic/eval_net/q/dense/bias/Adam
VariableV2*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ѓ
40/C_train/0/Critic/eval_net/q/dense/bias/Adam/AssignAssign-0/C_train/0/Critic/eval_net/q/dense/bias/Adam?0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
≈
20/C_train/0/Critic/eval_net/q/dense/bias/Adam/readIdentity-0/C_train/0/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ѕ
A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
ќ
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
µ
60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1A0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
…
40/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
_output_shapes
:*
T0
a
0/C_train/Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Y
0/C_train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Y
0/C_train/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
0/C_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
°
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_s(0/C_train/0/Critic/eval_net/l1/w1_s/Adam*0/C_train/0/Critic/eval_net/l1/w1_s/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonO0/C_train/gradients/0/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	С*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_s
Ґ
90/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/w1_a(0/C_train/0/Critic/eval_net/l1/w1_a/Adam*0/C_train/0/Critic/eval_net/l1/w1_a/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonQ0/C_train/gradients/0/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@0/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
Х
70/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam0/Critic/eval_net/l1/b1&0/C_train/0/Critic/eval_net/l1/b1/Adam(0/C_train/0/Critic/eval_net/l1/b1/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonN0/C_train/gradients/0/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:
»
@0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 0/Critic/eval_net/q/dense/kernel/0/C_train/0/Critic/eval_net/q/dense/kernel/Adam10/C_train/0/Critic/eval_net/q/dense/kernel/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonT0/C_train/gradients/0/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*3
_class)
'%loc:@0/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:
ї
>0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam0/Critic/eval_net/q/dense/bias-0/C_train/0/Critic/eval_net/q/dense/bias/Adam/0/C_train/0/Critic/eval_net/q/dense/bias/Adam_10/C_train/beta1_power/read0/C_train/beta2_power/read0/C_train/Adam/learning_rate0/C_train/Adam/beta10/C_train/Adam/beta20/C_train/Adam/epsilonU0/C_train/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*1
_class'
%#loc:@0/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
ќ
0/C_train/Adam/mulMul0/C_train/beta1_power/read0/C_train/Adam/beta18^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1
ј
0/C_train/Adam/AssignAssign0/C_train/beta1_power0/C_train/Adam/mul*
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
–
0/C_train/Adam/mul_1Mul0/C_train/beta2_power/read0/C_train/Adam/beta28^0/C_train/Adam/update_0/Critic/eval_net/l1/b1/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_a/ApplyAdam:^0/C_train/Adam/update_0/Critic/eval_net/l1/w1_s/ApplyAdam?^0/C_train/Adam/update_0/Critic/eval_net/q/dense/bias/ApplyAdamA^0/C_train/Adam/update_0/Critic/eval_net/q/dense/kernel/ApplyAdam**
_class 
loc:@0/Critic/eval_net/l1/b1*
_output_shapes
: *
T0
ƒ
0/C_train/Adam/Assign_1Assign0/C_train/beta2_power0/C_train/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@0/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
ю
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
 *  А?*
dtype0*
_output_shapes
: 
Ы
0/a_grad/gradients/FillFill0/a_grad/gradients/Shape0/a_grad/gradients/grad_ys_0*'
_output_shapes
:€€€€€€€€€*
T0*

index_type0
©
E0/a_grad/gradients/0/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0/a_grad/gradients/Fill*
data_formatNHWC*
_output_shapes
:*
T0
б
?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul0/a_grad/gradients/Fill%0/Critic/eval_net/q/dense/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(*
T0
ќ
A0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul0/Critic/eval_net/l1/Relu0/a_grad/gradients/Fill*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
‘
:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?0/a_grad/gradients/0/Critic/eval_net/q/dense/MatMul_grad/MatMul0/Critic/eval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€*
T0
Р
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ShapeShape0/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
Л
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ф
H0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Г
60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradH0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeReshape60/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
З
80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/Relu_grad/ReluGradJ0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ф
<0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape80/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Sum_1:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
С
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ShapeShape0/Critic/eval_net/l1/MatMul*
out_type0*
_output_shapes
:*
T0
Х
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1Shape0/Critic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0
О
F0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/SumSum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeF0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
с
80/a_grad/gradients/0/Critic/eval_net/l1/add_grad/ReshapeReshape40/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Г
60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_1Sum:0/a_grad/gradients/0/Critic/eval_net/l1/add_1_grad/ReshapeH0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1Reshape60/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Sum_180/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ъ
<0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_10/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(*
T0
к
>0/a_grad/gradients/0/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul0/Critic/StopGradient:0/a_grad/gradients/0/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
N
	0/mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
e
0/mul_8Mul	0/mul_8/x 0/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes
:	С
N
	0/mul_9/xConst*
_output_shapes
: *
valueB
 *
„#<*
dtype0
c
0/mul_9Mul	0/mul_9/x0/Critic/eval_net/l1/w1_s/read*
_output_shapes
:	С*
T0
J
0/add_4Add0/mul_80/mul_9*
T0*
_output_shapes
:	С
љ

0/Assign_4Assign0/Critic/target_net/l1/w1_s0/add_4*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С*
use_locking(
O

0/mul_10/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
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
„#<*
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
Љ

0/Assign_5Assign0/Critic/target_net/l1/w1_a0/add_5*
use_locking(*
T0*.
_class$
" loc:@0/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
O

0/mul_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
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
„#<*
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
0/add_6Add0/mul_120/mul_13*
T0*
_output_shapes

:
Є

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
 *§p}?*
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
„#<*
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
 

0/Assign_7Assign"0/Critic/target_net/q/dense/kernel0/add_7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@0/Critic/target_net/q/dense/kernel
O

0/mul_16/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
g
0/mul_16Mul
0/mul_16/x%0/Critic/target_net/q/dense/bias/read*
T0*
_output_shapes
:
O

0/mul_17/xConst*
_output_shapes
: *
valueB
 *
„#<*
dtype0
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
¬

0/Assign_8Assign 0/Critic/target_net/q/dense/bias0/add_8*
use_locking(*
T0*3
_class)
'%loc:@0/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
y
0/policy_grads/gradients/ShapeShape0/Actor/eval_net/a/scaled_a*
_output_shapes
:*
T0*
out_type0
g
"0/policy_grads/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
≠
0/policy_grads/gradients/FillFill0/policy_grads/gradients/Shape"0/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
Щ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeShape0/Actor/eval_net/a/truediv*
out_type0*
_output_shapes
:*
T0
Д
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
©
O0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ShapeA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulMul0/policy_grads/gradients/Fill0/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Ф
=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/SumSum=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/MulO0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
М
A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
≥
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Mul0/Actor/eval_net/a/truediv0/policy_grads/gradients/Fill*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Mul_1Q0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
C0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Sum_1A0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ч
>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/ShapeShape0/Actor/eval_net/a/a/Relu*
T0*
out_type0*
_output_shapes
:
Г
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
¶
N0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ё
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDivRealDivA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/Reshape0/Actor/eval_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
Х
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/SumSum@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDivN0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Й
@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/ReshapeReshape<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Р
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/NegNeg0/Actor/eval_net/a/a/Relu*'
_output_shapes
:€€€€€€€€€*
T0
џ
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_1RealDiv<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Neg0/Actor/eval_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
б
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_2RealDivB0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_10/Actor/eval_net/a/truediv/y*
T0*'
_output_shapes
:€€€€€€€€€
ь
<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/mulMulA0/policy_grads/gradients/0/Actor/eval_net/a/scaled_a_grad/ReshapeB0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:€€€€€€€€€
Х
>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum_1Sum<0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/mulP0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ю
B0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Reshape_1Reshape>0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Sum_1@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
џ
@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGradReluGrad@0/policy_grads/gradients/0/Actor/eval_net/a/truediv_grad/Reshape0/Actor/eval_net/a/a/Relu*
T0*'
_output_shapes
:€€€€€€€€€
”
F0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGrad@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ж
@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMulMatMul@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad 0/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b(
ч
B0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes

:d*
transpose_b( 
ў
?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad0/Actor/eval_net/l1/Tanh@0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€d
—
E0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Д
?0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad0/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€С*
transpose_b(
б
A0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?0/policy_grads/gradients/0/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes
:	Сd*
transpose_b( 
Ц
#0/A_train/beta1_power/initial_valueConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
І
0/A_train/beta1_power
VariableV2*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Џ
0/A_train/beta1_power/AssignAssign0/A_train/beta1_power#0/A_train/beta1_power/initial_value*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
М
0/A_train/beta1_power/readIdentity0/A_train/beta1_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
#0/A_train/beta2_power/initial_valueConst*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
І
0/A_train/beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@0/Actor/eval_net/a/a/bias
Џ
0/A_train/beta2_power/AssignAssign0/A_train/beta2_power#0/A_train/beta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
М
0/A_train/beta2_power/readIdentity0/A_train/beta2_power*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ћ
K0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"С  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
µ
A0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Њ
;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	Сd*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
ќ
)0/A_train/0/Actor/eval_net/l1/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	Сd
§
00/A_train/0/Actor/eval_net/l1/kernel/Adam/AssignAssign)0/A_train/0/Actor/eval_net/l1/kernel/Adam;0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
Њ
.0/A_train/0/Actor/eval_net/l1/kernel/Adam/readIdentity)0/A_train/0/Actor/eval_net/l1/kernel/Adam*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
Ќ
M0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"С  d   *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
Ј
C0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
dtype0
ƒ
=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
–
+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
	container *
shape:	Сd
™
20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1=0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
¬
00/A_train/0/Actor/eval_net/l1/kernel/Adam_1/readIdentity+0/A_train/0/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel
≥
90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:d*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias
ј
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
Ч
.0/A_train/0/Actor/eval_net/l1/bias/Adam/AssignAssign'0/A_train/0/Actor/eval_net/l1/bias/Adam90/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(
≥
,0/A_train/0/Actor/eval_net/l1/bias/Adam/readIdentity'0/A_train/0/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
µ
;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
valueBd*    *+
_class!
loc:@0/Actor/eval_net/l1/bias*
dtype0
¬
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
Э
00/A_train/0/Actor/eval_net/l1/bias/Adam_1/AssignAssign)0/A_train/0/Actor/eval_net/l1/bias/Adam_1;0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*+
_class!
loc:@0/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
Ј
.0/A_train/0/Actor/eval_net/l1/bias/Adam_1/readIdentity)0/A_train/0/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
_output_shapes
:d
Ѕ
<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:d*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
ќ
*0/A_train/0/Actor/eval_net/a/a/kernel/Adam
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
І
10/A_train/0/Actor/eval_net/a/a/kernel/Adam/AssignAssign*0/A_train/0/Actor/eval_net/a/a/kernel/Adam<0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
ј
/0/A_train/0/Actor/eval_net/a/a/kernel/Adam/readIdentity*0/A_train/0/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
_output_shapes

:d
√
>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:d*
valueBd*    *.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
dtype0
–
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
≠
30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1>0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
ƒ
10/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1*
_output_shapes

:d*
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel
µ
:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
¬
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
Ы
/0/A_train/0/Actor/eval_net/a/a/bias/Adam/AssignAssign(0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
ґ
-0/A_train/0/Actor/eval_net/a/a/bias/Adam/readIdentity(0/A_train/0/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
:
Ј
<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
ƒ
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
°
10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1<0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ї
/0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/readIdentity*0/A_train/0/Actor/eval_net/a/a/bias/Adam_1*
_output_shapes
:*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias
a
0/A_train/Adam/learning_rateConst*
valueB
 *oЇ*
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
 *wЊ?*
dtype0*
_output_shapes
: 
[
0/A_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Ш
:0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/kernel)0/A_train/0/Actor/eval_net/l1/kernel/Adam+0/A_train/0/Actor/eval_net/l1/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonA0/policy_grads/gradients/0/Actor/eval_net/l1/MatMul_grad/MatMul_1*
use_locking( *
T0*-
_class#
!loc:@0/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes
:	Сd
Н
80/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/l1/bias'0/A_train/0/Actor/eval_net/l1/bias/Adam)0/A_train/0/Actor/eval_net/l1/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonE0/policy_grads/gradients/0/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
use_locking( *
T0*+
_class!
loc:@0/Actor/eval_net/l1/bias*
use_nesterov( 
Э
;0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/kernel*0/A_train/0/Actor/eval_net/a/a/kernel/Adam,0/A_train/0/Actor/eval_net/a/a/kernel/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonB0/policy_grads/gradients/0/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
_output_shapes

:d*
use_locking( *
T0*.
_class$
" loc:@0/Actor/eval_net/a/a/kernel*
use_nesterov( 
У
90/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam0/Actor/eval_net/a/a/bias(0/A_train/0/Actor/eval_net/a/a/bias/Adam*0/A_train/0/Actor/eval_net/a/a/bias/Adam_10/A_train/beta1_power/read0/A_train/beta2_power/read0/A_train/Adam/learning_rate0/A_train/Adam/beta10/A_train/Adam/beta20/A_train/Adam/epsilonF0/policy_grads/gradients/0/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
М
0/A_train/Adam/mulMul0/A_train/beta1_power/read0/A_train/Adam/beta1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
¬
0/A_train/Adam/AssignAssign0/A_train/beta1_power0/A_train/Adam/mul*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(
О
0/A_train/Adam/mul_1Mul0/A_train/beta2_power/read0/A_train/Adam/beta2:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
_output_shapes
: 
∆
0/A_train/Adam/Assign_1Assign0/A_train/beta2_power0/A_train/Adam/mul_1*,
_class"
 loc:@0/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Ї
0/A_train/AdamNoOp^0/A_train/Adam/Assign^0/A_train/Adam/Assign_1:^0/A_train/Adam/update_0/Actor/eval_net/a/a/bias/ApplyAdam<^0/A_train/Adam/update_0/Actor/eval_net/a/a/kernel/ApplyAdam9^0/A_train/Adam/update_0/Actor/eval_net/l1/bias/ApplyAdam;^0/A_train/Adam/update_0/Actor/eval_net/l1/kernel/ApplyAdam
÷
initNoOp0^0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign2^0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign2^0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign4^0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign/^0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign1^0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign1^0/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign3^0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign^0/A_train/beta1_power/Assign^0/A_train/beta2_power/Assign!^0/Actor/eval_net/a/a/bias/Assign#^0/Actor/eval_net/a/a/kernel/Assign ^0/Actor/eval_net/l1/bias/Assign"^0/Actor/eval_net/l1/kernel/Assign#^0/Actor/target_net/a/a/bias/Assign%^0/Actor/target_net/a/a/kernel/Assign"^0/Actor/target_net/l1/bias/Assign$^0/Actor/target_net/l1/kernel/Assign.^0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign0^0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign0^0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign2^0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign5^0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign7^0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign7^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign9^0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign^0/C_train/beta1_power/Assign^0/C_train/beta2_power/Assign^0/Critic/eval_net/l1/b1/Assign!^0/Critic/eval_net/l1/w1_a/Assign!^0/Critic/eval_net/l1/w1_s/Assign&^0/Critic/eval_net/q/dense/bias/Assign(^0/Critic/eval_net/q/dense/kernel/Assign!^0/Critic/target_net/l1/b1/Assign#^0/Critic/target_net/l1/w1_a/Assign#^0/Critic/target_net/l1/w1_s/Assign(^0/Critic/target_net/q/dense/bias/Assign*^0/Critic/target_net/q/dense/kernel/Assign"&".
train_op"
 
0/C_train/Adam
0/A_train/Adam"Я9
	variablesС9О9
Ю
0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08
О
0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
Ґ
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
Т
0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08
§
0/Actor/target_net/l1/kernel:0#0/Actor/target_net/l1/kernel/Assign#0/Actor/target_net/l1/kernel/read:0280/Actor/target_net/l1/kernel/Initializer/random_normal:0
Ф
0/Actor/target_net/l1/bias:0!0/Actor/target_net/l1/bias/Assign!0/Actor/target_net/l1/bias/read:02.0/Actor/target_net/l1/bias/Initializer/Const:0
®
0/Actor/target_net/a/a/kernel:0$0/Actor/target_net/a/a/kernel/Assign$0/Actor/target_net/a/a/kernel/read:0290/Actor/target_net/a/a/kernel/Initializer/random_normal:0
Ш
0/Actor/target_net/a/a/bias:0"0/Actor/target_net/a/a/bias/Assign"0/Actor/target_net/a/a/bias/read:02/0/Actor/target_net/a/a/bias/Initializer/Const:0
Ъ
0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Ъ
0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
К
0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
ґ
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
¶
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:08
†
0/Critic/target_net/l1/w1_s:0"0/Critic/target_net/l1/w1_s/Assign"0/Critic/target_net/l1/w1_s/read:0270/Critic/target_net/l1/w1_s/Initializer/random_normal:0
†
0/Critic/target_net/l1/w1_a:0"0/Critic/target_net/l1/w1_a/Assign"0/Critic/target_net/l1/w1_a/read:0270/Critic/target_net/l1/w1_a/Initializer/random_normal:0
Р
0/Critic/target_net/l1/b1:0 0/Critic/target_net/l1/b1/Assign 0/Critic/target_net/l1/b1/read:02-0/Critic/target_net/l1/b1/Initializer/Const:0
Љ
$0/Critic/target_net/q/dense/kernel:0)0/Critic/target_net/q/dense/kernel/Assign)0/Critic/target_net/q/dense/kernel/read:02>0/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
ђ
"0/Critic/target_net/q/dense/bias:0'0/Critic/target_net/q/dense/bias/Assign'0/Critic/target_net/q/dense/bias/read:0240/Critic/target_net/q/dense/bias/Initializer/Const:0
|
0/C_train/beta1_power:00/C_train/beta1_power/Assign0/C_train/beta1_power/read:02%0/C_train/beta1_power/initial_value:0
|
0/C_train/beta2_power:00/C_train/beta2_power/Assign0/C_train/beta2_power/read:02%0/C_train/beta2_power/initial_value:0
ћ
*0/C_train/0/Critic/eval_net/l1/w1_s/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_s/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
‘
,0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
ћ
*0/C_train/0/Critic/eval_net/l1/w1_a/Adam:0/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Assign/0/C_train/0/Critic/eval_net/l1/w1_a/Adam/read:02<0/C_train/0/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
‘
,0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1:010/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Assign10/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/read:02>0/C_train/0/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
ƒ
(0/C_train/0/Critic/eval_net/l1/b1/Adam:0-0/C_train/0/Critic/eval_net/l1/b1/Adam/Assign-0/C_train/0/Critic/eval_net/l1/b1/Adam/read:02:0/C_train/0/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
ћ
*0/C_train/0/Critic/eval_net/l1/b1/Adam_1:0/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Assign/0/C_train/0/Critic/eval_net/l1/b1/Adam_1/read:02<0/C_train/0/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
и
10/C_train/0/Critic/eval_net/q/dense/kernel/Adam:060/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Assign60/C_train/0/Critic/eval_net/q/dense/kernel/Adam/read:02C0/C_train/0/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
р
30/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1:080/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Assign80/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/read:02E0/C_train/0/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
а
/0/C_train/0/Critic/eval_net/q/dense/bias/Adam:040/C_train/0/Critic/eval_net/q/dense/bias/Adam/Assign40/C_train/0/Critic/eval_net/q/dense/bias/Adam/read:02A0/C_train/0/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
и
10/C_train/0/Critic/eval_net/q/dense/bias/Adam_1:060/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Assign60/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/read:02C0/C_train/0/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
0/A_train/beta1_power:00/A_train/beta1_power/Assign0/A_train/beta1_power/read:02%0/A_train/beta1_power/initial_value:0
|
0/A_train/beta2_power:00/A_train/beta2_power/Assign0/A_train/beta2_power/read:02%0/A_train/beta2_power/initial_value:0
–
+0/A_train/0/Actor/eval_net/l1/kernel/Adam:000/A_train/0/Actor/eval_net/l1/kernel/Adam/Assign00/A_train/0/Actor/eval_net/l1/kernel/Adam/read:02=0/A_train/0/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
Ў
-0/A_train/0/Actor/eval_net/l1/kernel/Adam_1:020/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Assign20/A_train/0/Actor/eval_net/l1/kernel/Adam_1/read:02?0/A_train/0/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
»
)0/A_train/0/Actor/eval_net/l1/bias/Adam:0.0/A_train/0/Actor/eval_net/l1/bias/Adam/Assign.0/A_train/0/Actor/eval_net/l1/bias/Adam/read:02;0/A_train/0/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
–
+0/A_train/0/Actor/eval_net/l1/bias/Adam_1:000/A_train/0/Actor/eval_net/l1/bias/Adam_1/Assign00/A_train/0/Actor/eval_net/l1/bias/Adam_1/read:02=0/A_train/0/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
‘
,0/A_train/0/Actor/eval_net/a/a/kernel/Adam:010/A_train/0/Actor/eval_net/a/a/kernel/Adam/Assign10/A_train/0/Actor/eval_net/a/a/kernel/Adam/read:02>0/A_train/0/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
№
.0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1:030/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Assign30/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/read:02@0/A_train/0/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
ћ
*0/A_train/0/Actor/eval_net/a/a/bias/Adam:0/0/A_train/0/Actor/eval_net/a/a/bias/Adam/Assign/0/A_train/0/Actor/eval_net/a/a/bias/Adam/read:02<0/A_train/0/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
‘
,0/A_train/0/Actor/eval_net/a/a/bias/Adam_1:010/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Assign10/A_train/0/Actor/eval_net/a/a/bias/Adam_1/read:02>0/A_train/0/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0"∞
trainable_variablesШХ
Ю
0/Actor/eval_net/l1/kernel:0!0/Actor/eval_net/l1/kernel/Assign!0/Actor/eval_net/l1/kernel/read:0260/Actor/eval_net/l1/kernel/Initializer/random_normal:08
О
0/Actor/eval_net/l1/bias:00/Actor/eval_net/l1/bias/Assign0/Actor/eval_net/l1/bias/read:02,0/Actor/eval_net/l1/bias/Initializer/Const:08
Ґ
0/Actor/eval_net/a/a/kernel:0"0/Actor/eval_net/a/a/kernel/Assign"0/Actor/eval_net/a/a/kernel/read:0270/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
Т
0/Actor/eval_net/a/a/bias:0 0/Actor/eval_net/a/a/bias/Assign 0/Actor/eval_net/a/a/bias/read:02-0/Actor/eval_net/a/a/bias/Initializer/Const:08
Ъ
0/Critic/eval_net/l1/w1_s:0 0/Critic/eval_net/l1/w1_s/Assign 0/Critic/eval_net/l1/w1_s/read:0250/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Ъ
0/Critic/eval_net/l1/w1_a:0 0/Critic/eval_net/l1/w1_a/Assign 0/Critic/eval_net/l1/w1_a/read:0250/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
К
0/Critic/eval_net/l1/b1:00/Critic/eval_net/l1/b1/Assign0/Critic/eval_net/l1/b1/read:02+0/Critic/eval_net/l1/b1/Initializer/Const:08
ґ
"0/Critic/eval_net/q/dense/kernel:0'0/Critic/eval_net/q/dense/kernel/Assign'0/Critic/eval_net/q/dense/kernel/read:02<0/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
¶
 0/Critic/eval_net/q/dense/bias:0%0/Critic/eval_net/q/dense/bias/Assign%0/Critic/eval_net/q/dense/bias/read:0220/Critic/eval_net/q/dense/bias/Initializer/Const:085д@п