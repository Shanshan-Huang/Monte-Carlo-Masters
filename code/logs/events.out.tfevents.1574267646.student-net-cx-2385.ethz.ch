       £K"	  АњZu„Abrain.Event:2ћU≈"Нx     
Љ.	'•њZu„A"Ас
h
S/sPlaceholder*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С*
dtype0
f
R/rPlaceholder*
shape:€€€€€€€€€*
dtype0*'
_output_shapes
:€€€€€€€€€
j
S_/s_Placeholder*
dtype0*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С
Ї
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"С  d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
≠
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *ЌћL=*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
ѓ
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0
Ь
I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*

seed*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes
:	Сd
†
81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
Й
41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
њ
1/Actor/eval_net/l1/kernel
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd
€
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
†
1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
§
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*
valueBd*  А?*+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
±
1/Actor/eval_net/l1/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container 
к
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Х
1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
Ґ
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
™
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
Љ
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
:
ѓ
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *ЌћL=*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes
: 
±
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
valueB
 *   ?*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0
Ю
J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:d*

seed*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
£
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
_output_shapes

:d*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
М
51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
њ
1/Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d
В
"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(
Ґ
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
¶
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*
valueB*  А?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
≥
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
о
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ш
1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
є
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
≠
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
b
1/Actor/eval_net/a/scaled_a/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
С
1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Њ
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*
valueB"С  d   */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
:
±
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*
valueB
 *ЌћL=*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
≥
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
dtype0*
_output_shapes
: 
Ґ
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
seed2(*
dtype0*
_output_shapes
:	Сd*

seed*
T0
®
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
С
61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
√
1/Actor/target_net/l1/kernel
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container *
shape:	Сd
З
#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
¶
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
®
,1/Actor/target_net/l1/bias/Initializer/ConstConst*
valueBd*  А?*-
_class#
!loc:@1/Actor/target_net/l1/bias*
dtype0*
_output_shapes
:d
µ
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
т
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ы
1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*-
_class#
!loc:@1/Actor/target_net/l1/bias*
_output_shapes
:d*
T0
®
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
∞
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*'
_output_shapes
:€€€€€€€€€d*
T0
ј
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*
valueB"d      *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
:
≥
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*
valueB
 *ЌћL=*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
µ
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
dtype0*
_output_shapes
: 
§
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:d*

seed*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed28
Ђ
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0
Ф
71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel
√
1/Actor/target_net/a/a/kernel
VariableV2*
_output_shapes

:d*
shared_name *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container *
shape
:d*
dtype0
К
$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
®
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d*
T0
™
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*
valueB*  А?*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
dtype0*
_output_shapes
:
Ј
1/Actor/target_net/a/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@1/Actor/target_net/a/a/bias
ц
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ю
 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
_output_shapes
:
њ
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
≥
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
d
1/Actor/target_net/a/scaled_a/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
L
1/mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
b
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
_output_shapes
:	Сd*
T0
N
	1/mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
d
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
T0*
_output_shapes
:	Сd
F
1/addAdd1/mul1/mul_1*
_output_shapes
:	Сd*
T0
ї
1/AssignAssign1/Actor/target_net/l1/kernel1/add*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
N
	1/mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:d
N
	1/mul_3/xConst*
valueB
 *
„#<*
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
ґ

1/Assign_1Assign1/Actor/target_net/l1/bias1/add_1*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(*
_output_shapes
:d
N
	1/mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *§p}?
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
_output_shapes

:d*
T0
N
	1/mul_5/xConst*
_output_shapes
: *
valueB
 *
„#<*
dtype0
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
ј

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
 *§p}?*
dtype0*
_output_shapes
: 
`
1/mul_6Mul	1/mul_6/x 1/Actor/target_net/a/a/bias/read*
T0*
_output_shapes
:
N
	1/mul_7/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
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
Є

1/Assign_3Assign1/Actor/target_net/a/a/bias1/add_3*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
1/Critic/StopGradientStopGradient1/Actor/eval_net/a/scaled_a*'
_output_shapes
:€€€€€€€€€*
T0
Є
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"С     *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
Ђ
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
≠
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Щ
H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
seed2c*
dtype0*
_output_shapes
:	С*

seed
Ь
71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes
:	С*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Е
31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
_output_shapes
:	С*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
љ
1/Critic/eval_net/l1/w1_s
VariableV2*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С
ы
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С*
use_locking(*
T0
Э
1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Є
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes
:
Ђ
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0
≠
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Ш
H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
seed2l
Ы
71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Д
31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
ї
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
ъ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(
Ь
1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
™
)1/Critic/eval_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*
valueB*  А?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0
Ј
1/Critic/eval_net/l1/b1
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
к
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Ц
1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1*
_output_shapes

:*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Ґ
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
ґ
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Н
1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*
T0*'
_output_shapes
:€€€€€€€€€
Л
1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€*
T0
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
∆
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
є
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
ї
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes
: 
≠
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*
seed2~*
dtype0*
_output_shapes

:*

seed*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
Ј
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
†
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
…
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
Ц
'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
±
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
∞
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*
valueB*  А?*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
љ
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
В
%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(
І
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
ƒ
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Љ
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Љ
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*
valueB"С     *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
:
ѓ
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
±
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
dtype0*
_output_shapes
: 
†
J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	С*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
seed2Н
§
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
_output_shapes
:	С*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s
Н
51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Ѕ
1/Critic/target_net/l1/w1_s
VariableV2*
	container *
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_s
Г
"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С*
use_locking(*
T0
£
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*
_output_shapes
:	С*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s
Љ
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
valueB"      *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
:
ѓ
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*
valueB
 *    *.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
±
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*
valueB
 *   ?*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
dtype0*
_output_shapes
: 
Я
J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2Ц*
dtype0*
_output_shapes

:*

seed
£
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
М
51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
њ
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
В
"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ґ
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:
Ѓ
+1/Critic/target_net/l1/b1/Initializer/ConstConst*
valueB*  А?*,
_class"
 loc:@1/Critic/target_net/l1/b1*
dtype0*
_output_shapes

:
ї
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
т
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(*
_output_shapes

:
Ь
1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1
®
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
¬
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
У
1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*
T0*'
_output_shapes
:€€€€€€€€€
С
1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*'
_output_shapes
:€€€€€€€€€*
T0
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
 
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*
valueB"      *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
:
љ
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*
valueB
 *    *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
њ
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*
valueB
 *   ?*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
dtype0*
_output_shapes
: 
і
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2®
њ
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
®
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
Ќ
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
Ю
)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*
use_locking(*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
Ј
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:
і
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*
valueB*  А?*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
dtype0*
_output_shapes
:
Ѕ
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
К
'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
≠
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
_output_shapes
:*
T0
 
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
¬
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*'
_output_shapes
:€€€€€€€€€*
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
1/target_q/mulMul1/target_q/mul/x#1/Critic/target_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
\
1/target_q/addAddR/r1/target_q/mul*
T0*'
_output_shapes
:€€€€€€€€€
Ц
1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
a
1/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
1/TD_error/MeanMean1/TD_error/SquaredDifference1/TD_error/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
\
1/C_train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
b
1/C_train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
Н
1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
З
61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
ƒ
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
К
.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
џ
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
М
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
’
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
ў
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
Ѕ
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
њ
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
®
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Ћ
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Й
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
T0*
out_type0*
_output_shapes
:
Ю
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
в
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*'
_output_shapes
:€€€€€€€€€*
T0
Ў
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
к
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*'
_output_shapes
:€€€€€€€€€*
T0
К
91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
А
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
О
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
≥
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
 
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
ж
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
а
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
г
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
п
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
€
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€*
T0
ч
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ю
@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
Л
B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
Џ
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
ф
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
с
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
и
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€*
T0
С
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
T0*
out_type0*
_output_shapes
:
М
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB"      *
dtype0
Ч
I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ж
71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ъ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
К
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
 
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
ё
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
џ
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:
Т
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Ц
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
_output_shapes
:*
T0*
out_type0
С
G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
У
51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ф
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Ч
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ъ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ƒ
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
÷
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape
№
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
К
;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*
T0*(
_output_shapes
:€€€€€€€€€С*
transpose_a( *
transpose_b(
и
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	С*
transpose_a(*
transpose_b( 
Ћ
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
б
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€С
ё
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	С
Н
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
э
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
—
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
и
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:€€€€€€€€€
е
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
Ф
#1/C_train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
•
1/C_train/beta1_power
VariableV2*
_output_shapes
: *
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0
Ў
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: 
К
1/C_train/beta1_power/readIdentity1/C_train/beta1_power*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
Ф
#1/C_train/beta2_power/initial_valueConst*
valueB
 *wЊ?**
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes
: 
•
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
Ў
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(
К
1/C_train/beta2_power/readIdentity1/C_train/beta2_power*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
…
J1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
≥
@1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
_output_shapes
:	С*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*

index_type0
ћ
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape:	С
†
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
ї
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Ћ
L1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
µ
B1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*

index_type0*
_output_shapes
:	С
ќ
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape:	С
¶
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
њ
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С*
T0
љ
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
 
(1/C_train/1/Critic/eval_net/l1/w1_a/Adam
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
Я
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ї
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
њ
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB*    *
dtype0*
_output_shapes

:
ћ
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
•
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
Њ
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
є
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    *
dtype0
∆
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
Ч
-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
і
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
ї
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
»
(1/C_train/1/Critic/eval_net/l1/b1/Adam_1
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
Э
/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/AssignAssign(1/C_train/1/Critic/eval_net/l1/b1/Adam_1:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
Є
-1/C_train/1/Critic/eval_net/l1/b1/Adam_1/readIdentity(1/C_train/1/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ћ
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    *
dtype0*
_output_shapes

:
Ў
/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam
VariableV2*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
ї
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
ѕ
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
Ќ
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB*    
Џ
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
Ѕ
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
”
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
њ
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
ћ
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
ѓ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
≈
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
Ѕ
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
ќ
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
µ
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
…
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*
_output_shapes
:*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
a
1/C_train/Adam/learning_rateConst*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
Y
1/C_train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Y
1/C_train/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
°
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	С*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( 
Ґ
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
Х
71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1**
_class 
loc:@1/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
»
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( 
ї
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
ќ
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
ј
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
–
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
ƒ
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
ю
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
 *  А?*
dtype0*
_output_shapes
: 
Ы
1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
©
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
б
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
ќ
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
‘
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
Р
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
Л
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ф
H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Г
61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ч
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
З
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
С
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
Х
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
О
F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
с
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Г
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ч
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ъ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
к
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
N
	1/mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
e
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
T0*
_output_shapes
:	С
N
	1/mul_9/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
c
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
T0*
_output_shapes
:	С
J
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes
:	С
љ

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
O

1/mul_10/xConst*
valueB
 *§p}?*
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
„#<*
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
Љ

1/Assign_5Assign1/Critic/target_net/l1/w1_a1/add_5*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(
O

1/mul_12/xConst*
valueB
 *§p}?*
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
„#<*
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
Є

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
 *§p}?*
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
„#<*
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
1/add_7Add1/mul_141/mul_15*
T0*
_output_shapes

:
 

1/Assign_7Assign"1/Critic/target_net/q/dense/kernel1/add_7*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
O

1/mul_16/xConst*
_output_shapes
: *
valueB
 *§p}?*
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
„#<*
dtype0*
_output_shapes
: 
e
1/mul_17Mul
1/mul_17/x#1/Critic/eval_net/q/dense/bias/read*
_output_shapes
:*
T0
G
1/add_8Add1/mul_161/mul_17*
T0*
_output_shapes
:
¬

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
 *  А?*
dtype0*
_output_shapes
: 
≠
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*'
_output_shapes
:€€€€€€€€€*
T0*

index_type0
Ы
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
_output_shapes
:*
T0*
out_type0
Д
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
©
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Ф
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
µ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
и
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
ў
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
М
@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(*
T0
э
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0
ў
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*
T0*'
_output_shapes
:€€€€€€€€€d
—
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Д
?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€С*
transpose_a( *
transpose_b(
б
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
_output_shapes
:	Сd*
transpose_a(*
transpose_b( *
T0
Ц
#1/A_train/beta1_power/initial_valueConst*
valueB
 *fff?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
: 
І
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
Џ
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
М
1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
#1/A_train/beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wЊ?*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0
І
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
Џ
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
М
1/A_train/beta2_power/readIdentity1/A_train/beta2_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ћ
K1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
µ
A1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes
:	Сd
ќ
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd
§
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
Њ
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Ќ
M1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
Ј
C1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ƒ
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*

index_type0*
_output_shapes
:	Сd
–
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
shape:	Сd*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container 
™
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
¬
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
≥
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*    *
dtype0*
_output_shapes
:d
ј
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
Ч
.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
≥
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
µ
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:d*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*    
¬
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
Э
01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Ј
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
Ѕ
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
_output_shapes

:d*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0
ќ
*1/A_train/1/Actor/eval_net/a/a/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d
І
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
ј
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
√
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueBd*    *
dtype0*
_output_shapes

:d
–
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
≠
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(
ƒ
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
µ
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    *
dtype0*
_output_shapes
:
¬
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
Ы
/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
ґ
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
Ј
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*    
ƒ
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
°
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ї
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
a
1/A_train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *Ј—Є
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
 *wЊ?*
dtype0*
_output_shapes
: 
[
1/A_train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
Ш
:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	Сd*
use_locking( *
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
use_nesterov( 
Н
81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d*
use_locking( *
T0
Э
;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d
У
91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:
М
1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
¬
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
О
1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: *
T0
∆
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(
Ї
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
÷
initNoOp0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&ЎФм«3І     ћі≤ј	М¶њZu„AJ¶ќ
Ъу
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
shared_namestring И*1.14.02v1.14.0-rc1-22-gaf24dc91b5Ас
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
S_/s_Placeholder*
dtype0*(
_output_shapes
:€€€€€€€€€С*
shape:€€€€€€€€€С
Ї
:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shapeConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
≠
91/Actor/eval_net/l1/kernel/Initializer/random_normal/meanConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
ѓ
;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddevConst*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ь
I1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal:1/Actor/eval_net/l1/kernel/Initializer/random_normal/shape*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
seed2*
dtype0*
_output_shapes
:	Сd*

seed*
T0
†
81/Actor/eval_net/l1/kernel/Initializer/random_normal/mulMulI1/Actor/eval_net/l1/kernel/Initializer/random_normal/RandomStandardNormal;1/Actor/eval_net/l1/kernel/Initializer/random_normal/stddev*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
Й
41/Actor/eval_net/l1/kernel/Initializer/random_normalAdd81/Actor/eval_net/l1/kernel/Initializer/random_normal/mul91/Actor/eval_net/l1/kernel/Initializer/random_normal/mean*
_output_shapes
:	Сd*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel
њ
1/Actor/eval_net/l1/kernel
VariableV2*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape:	Сd*
dtype0*
_output_shapes
:	Сd
€
!1/Actor/eval_net/l1/kernel/AssignAssign1/Actor/eval_net/l1/kernel41/Actor/eval_net/l1/kernel/Initializer/random_normal*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
†
1/Actor/eval_net/l1/kernel/readIdentity1/Actor/eval_net/l1/kernel*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
§
*1/Actor/eval_net/l1/bias/Initializer/ConstConst*+
_class!
loc:@1/Actor/eval_net/l1/bias*
valueBd*  А?*
dtype0*
_output_shapes
:d
±
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
к
1/Actor/eval_net/l1/bias/AssignAssign1/Actor/eval_net/l1/bias*1/Actor/eval_net/l1/bias/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
Х
1/Actor/eval_net/l1/bias/readIdentity1/Actor/eval_net/l1/bias*
_output_shapes
:d*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
Ґ
1/Actor/eval_net/l1/MatMulMatMulS/s1/Actor/eval_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b( *
T0
™
1/Actor/eval_net/l1/BiasAddBiasAdd1/Actor/eval_net/l1/MatMul1/Actor/eval_net/l1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
o
1/Actor/eval_net/l1/TanhTanh1/Actor/eval_net/l1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
Љ
;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB"d      *
dtype0
ѓ
:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
±
<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ю
J1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Actor/eval_net/a/a/kernel/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
seed2*
dtype0*
_output_shapes

:d*

seed
£
91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mulMulJ1/Actor/eval_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal<1/Actor/eval_net/a/a/kernel/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
М
51/Actor/eval_net/a/a/kernel/Initializer/random_normalAdd91/Actor/eval_net/a/a/kernel/Initializer/random_normal/mul:1/Actor/eval_net/a/a/kernel/Initializer/random_normal/mean*
_output_shapes

:d*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
њ
1/Actor/eval_net/a/a/kernel
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d
В
"1/Actor/eval_net/a/a/kernel/AssignAssign1/Actor/eval_net/a/a/kernel51/Actor/eval_net/a/a/kernel/Initializer/random_normal*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel
Ґ
 1/Actor/eval_net/a/a/kernel/readIdentity1/Actor/eval_net/a/a/kernel*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
¶
+1/Actor/eval_net/a/a/bias/Initializer/ConstConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB*  А?*
dtype0*
_output_shapes
:
≥
1/Actor/eval_net/a/a/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias
о
 1/Actor/eval_net/a/a/bias/AssignAssign1/Actor/eval_net/a/a/bias+1/Actor/eval_net/a/a/bias/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ш
1/Actor/eval_net/a/a/bias/readIdentity1/Actor/eval_net/a/a/bias*
_output_shapes
:*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
є
1/Actor/eval_net/a/a/MatMulMatMul1/Actor/eval_net/l1/Tanh 1/Actor/eval_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
≠
1/Actor/eval_net/a/a/BiasAddBiasAdd1/Actor/eval_net/a/a/MatMul1/Actor/eval_net/a/a/bias/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
w
1/Actor/eval_net/a/a/SigmoidSigmoid1/Actor/eval_net/a/a/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
b
1/Actor/eval_net/a/scaled_a/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
С
1/Actor/eval_net/a/scaled_aMul1/Actor/eval_net/a/a/Sigmoid1/Actor/eval_net/a/scaled_a/y*
T0*'
_output_shapes
:€€€€€€€€€
Њ
<1/Actor/target_net/l1/kernel/Initializer/random_normal/shapeConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB"С  d   *
dtype0*
_output_shapes
:
±
;1/Actor/target_net/l1/kernel/Initializer/random_normal/meanConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
≥
=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddevConst*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ґ
K1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal<1/Actor/target_net/l1/kernel/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	Сd*

seed*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
seed2(
®
:1/Actor/target_net/l1/kernel/Initializer/random_normal/mulMulK1/Actor/target_net/l1/kernel/Initializer/random_normal/RandomStandardNormal=1/Actor/target_net/l1/kernel/Initializer/random_normal/stddev*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd*
T0
С
61/Actor/target_net/l1/kernel/Initializer/random_normalAdd:1/Actor/target_net/l1/kernel/Initializer/random_normal/mul;1/Actor/target_net/l1/kernel/Initializer/random_normal/mean*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
√
1/Actor/target_net/l1/kernel
VariableV2*
shape:	Сd*
dtype0*
_output_shapes
:	Сd*
shared_name */
_class%
#!loc:@1/Actor/target_net/l1/kernel*
	container 
З
#1/Actor/target_net/l1/kernel/AssignAssign1/Actor/target_net/l1/kernel61/Actor/target_net/l1/kernel/Initializer/random_normal*
_output_shapes
:	Сd*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
validate_shape(
¶
!1/Actor/target_net/l1/kernel/readIdentity1/Actor/target_net/l1/kernel*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel*
_output_shapes
:	Сd
®
,1/Actor/target_net/l1/bias/Initializer/ConstConst*-
_class#
!loc:@1/Actor/target_net/l1/bias*
valueBd*  А?*
dtype0*
_output_shapes
:d
µ
1/Actor/target_net/l1/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@1/Actor/target_net/l1/bias*
	container 
т
!1/Actor/target_net/l1/bias/AssignAssign1/Actor/target_net/l1/bias,1/Actor/target_net/l1/bias/Initializer/Const*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
validate_shape(
Ы
1/Actor/target_net/l1/bias/readIdentity1/Actor/target_net/l1/bias*
T0*-
_class#
!loc:@1/Actor/target_net/l1/bias*
_output_shapes
:d
®
1/Actor/target_net/l1/MatMulMatMulS_/s_!1/Actor/target_net/l1/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b( *
T0
∞
1/Actor/target_net/l1/BiasAddBiasAdd1/Actor/target_net/l1/MatMul1/Actor/target_net/l1/bias/read*'
_output_shapes
:€€€€€€€€€d*
T0*
data_formatNHWC
s
1/Actor/target_net/l1/TanhTanh1/Actor/target_net/l1/BiasAdd*'
_output_shapes
:€€€€€€€€€d*
T0
ј
=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shapeConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB"d      *
dtype0*
_output_shapes
:
≥
<1/Actor/target_net/a/a/kernel/Initializer/random_normal/meanConst*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
µ
>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
valueB
 *   ?
§
L1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=1/Actor/target_net/a/a/kernel/Initializer/random_normal/shape*

seed*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
seed28*
dtype0*
_output_shapes

:d
Ђ
;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mulMulL1/Actor/target_net/a/a/kernel/Initializer/random_normal/RandomStandardNormal>1/Actor/target_net/a/a/kernel/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
Ф
71/Actor/target_net/a/a/kernel/Initializer/random_normalAdd;1/Actor/target_net/a/a/kernel/Initializer/random_normal/mul<1/Actor/target_net/a/a/kernel/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
√
1/Actor/target_net/a/a/kernel
VariableV2*
shape
:d*
dtype0*
_output_shapes

:d*
shared_name *0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
	container 
К
$1/Actor/target_net/a/a/kernel/AssignAssign1/Actor/target_net/a/a/kernel71/Actor/target_net/a/a/kernel/Initializer/random_normal*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(*
_output_shapes

:d*
use_locking(*
T0
®
"1/Actor/target_net/a/a/kernel/readIdentity1/Actor/target_net/a/a/kernel*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
_output_shapes

:d
™
-1/Actor/target_net/a/a/bias/Initializer/ConstConst*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
valueB*  А?*
dtype0*
_output_shapes
:
Ј
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
ц
"1/Actor/target_net/a/a/bias/AssignAssign1/Actor/target_net/a/a/bias-1/Actor/target_net/a/a/bias/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ю
 1/Actor/target_net/a/a/bias/readIdentity1/Actor/target_net/a/a/bias*
T0*.
_class$
" loc:@1/Actor/target_net/a/a/bias*
_output_shapes
:
њ
1/Actor/target_net/a/a/MatMulMatMul1/Actor/target_net/l1/Tanh"1/Actor/target_net/a/a/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
≥
1/Actor/target_net/a/a/BiasAddBiasAdd1/Actor/target_net/a/a/MatMul 1/Actor/target_net/a/a/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
{
1/Actor/target_net/a/a/SigmoidSigmoid1/Actor/target_net/a/a/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
d
1/Actor/target_net/a/scaled_a/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
Ч
1/Actor/target_net/a/scaled_aMul1/Actor/target_net/a/a/Sigmoid1/Actor/target_net/a/scaled_a/y*'
_output_shapes
:€€€€€€€€€*
T0
L
1/mul/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
b
1/mulMul1/mul/x!1/Actor/target_net/l1/kernel/read*
T0*
_output_shapes
:	Сd
N
	1/mul_1/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
d
1/mul_1Mul	1/mul_1/x1/Actor/eval_net/l1/kernel/read*
_output_shapes
:	Сd*
T0
F
1/addAdd1/mul1/mul_1*
_output_shapes
:	Сd*
T0
ї
1/AssignAssign1/Actor/target_net/l1/kernel1/add*
validate_shape(*
_output_shapes
:	Сd*
use_locking(*
T0*/
_class%
#!loc:@1/Actor/target_net/l1/kernel
N
	1/mul_2/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
_
1/mul_2Mul	1/mul_2/x1/Actor/target_net/l1/bias/read*
T0*
_output_shapes
:d
N
	1/mul_3/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
]
1/mul_3Mul	1/mul_3/x1/Actor/eval_net/l1/bias/read*
T0*
_output_shapes
:d
E
1/add_1Add1/mul_21/mul_3*
_output_shapes
:d*
T0
ґ

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
 *§p}?*
dtype0*
_output_shapes
: 
f
1/mul_4Mul	1/mul_4/x"1/Actor/target_net/a/a/kernel/read*
T0*
_output_shapes

:d
N
	1/mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
d
1/mul_5Mul	1/mul_5/x 1/Actor/eval_net/a/a/kernel/read*
T0*
_output_shapes

:d
I
1/add_2Add1/mul_41/mul_5*
T0*
_output_shapes

:d
ј

1/Assign_2Assign1/Actor/target_net/a/a/kernel1/add_2*
_output_shapes

:d*
use_locking(*
T0*0
_class&
$"loc:@1/Actor/target_net/a/a/kernel*
validate_shape(
N
	1/mul_6/xConst*
valueB
 *§p}?*
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
„#<*
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
Є

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
:€€€€€€€€€*
T0
Є
91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
Ђ
81/Critic/eval_net/l1/w1_s/Initializer/random_normal/meanConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
valueB
 *   ?
Щ
H1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_s/Initializer/random_normal/shape*
dtype0*
_output_shapes
:	С*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
seed2c
Ь
71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_s/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Е
31/Critic/eval_net/l1/w1_s/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_s/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_s/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
љ
1/Critic/eval_net/l1/w1_s
VariableV2*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0
ы
 1/Critic/eval_net/l1/w1_s/AssignAssign1/Critic/eval_net/l1/w1_s31/Critic/eval_net/l1/w1_s/Initializer/random_normal*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С*
use_locking(*
T0
Э
1/Critic/eval_net/l1/w1_s/readIdentity1/Critic/eval_net/l1/w1_s*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
Є
91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shapeConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB"      *
dtype0*
_output_shapes
:
Ђ
81/Critic/eval_net/l1/w1_a/Initializer/random_normal/meanConst*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
≠
:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
valueB
 *   ?
Ш
H1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal91/Critic/eval_net/l1/w1_a/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
seed2l
Ы
71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mulMulH1/Critic/eval_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal:1/Critic/eval_net/l1/w1_a/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
Д
31/Critic/eval_net/l1/w1_a/Initializer/random_normalAdd71/Critic/eval_net/l1/w1_a/Initializer/random_normal/mul81/Critic/eval_net/l1/w1_a/Initializer/random_normal/mean*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
ї
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
ъ
 1/Critic/eval_net/l1/w1_a/AssignAssign1/Critic/eval_net/l1/w1_a31/Critic/eval_net/l1/w1_a/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Ь
1/Critic/eval_net/l1/w1_a/readIdentity1/Critic/eval_net/l1/w1_a*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
™
)1/Critic/eval_net/l1/b1/Initializer/ConstConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB*  А?*
dtype0*
_output_shapes

:
Ј
1/Critic/eval_net/l1/b1
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
к
1/Critic/eval_net/l1/b1/AssignAssign1/Critic/eval_net/l1/b1)1/Critic/eval_net/l1/b1/Initializer/Const*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Ц
1/Critic/eval_net/l1/b1/readIdentity1/Critic/eval_net/l1/b1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ґ
1/Critic/eval_net/l1/MatMulMatMulS/s1/Critic/eval_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
ґ
1/Critic/eval_net/l1/MatMul_1MatMul1/Critic/StopGradient1/Critic/eval_net/l1/w1_a/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
Н
1/Critic/eval_net/l1/addAdd1/Critic/eval_net/l1/MatMul1/Critic/eval_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
Л
1/Critic/eval_net/l1/add_1Add1/Critic/eval_net/l1/add1/Critic/eval_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
o
1/Critic/eval_net/l1/ReluRelu1/Critic/eval_net/l1/add_1*
T0*'
_output_shapes
:€€€€€€€€€
∆
@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB"      
є
?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/meanConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ї
A1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddevConst*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
≠
O1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal@1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/shape*

seed*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
seed2~*
dtype0*
_output_shapes

:
Ј
>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mulMulO1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalA1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/stddev*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
†
:1/Critic/eval_net/q/dense/kernel/Initializer/random_normalAdd>1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mul?1/Critic/eval_net/q/dense/kernel/Initializer/random_normal/mean*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
…
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
Ц
'1/Critic/eval_net/q/dense/kernel/AssignAssign 1/Critic/eval_net/q/dense/kernel:1/Critic/eval_net/q/dense/kernel/Initializer/random_normal*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
±
%1/Critic/eval_net/q/dense/kernel/readIdentity 1/Critic/eval_net/q/dense/kernel*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
_output_shapes

:
∞
01/Critic/eval_net/q/dense/bias/Initializer/ConstConst*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
valueB*  А?*
dtype0*
_output_shapes
:
љ
1/Critic/eval_net/q/dense/bias
VariableV2*
shared_name *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
В
%1/Critic/eval_net/q/dense/bias/AssignAssign1/Critic/eval_net/q/dense/bias01/Critic/eval_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
І
#1/Critic/eval_net/q/dense/bias/readIdentity1/Critic/eval_net/q/dense/bias*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
_output_shapes
:
ƒ
 1/Critic/eval_net/q/dense/MatMulMatMul1/Critic/eval_net/l1/Relu%1/Critic/eval_net/q/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
Љ
!1/Critic/eval_net/q/dense/BiasAddBiasAdd 1/Critic/eval_net/q/dense/MatMul#1/Critic/eval_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Љ
;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shapeConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB"С     *
dtype0*
_output_shapes
:
ѓ
:1/Critic/target_net/l1/w1_s/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *    *
dtype0*
_output_shapes
: 
±
<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
valueB
 *   ?
†
J1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_s/Initializer/random_normal/shape*

seed*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
seed2Н*
dtype0*
_output_shapes
:	С
§
91/Critic/target_net/l1/w1_s/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_s/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_s/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Н
51/Critic/target_net/l1/w1_s/Initializer/random_normalAdd91/Critic/target_net/l1/w1_s/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_s/Initializer/random_normal/mean*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes
:	С
Ѕ
1/Critic/target_net/l1/w1_s
VariableV2*
dtype0*
_output_shapes
:	С*
shared_name *.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
	container *
shape:	С
Г
"1/Critic/target_net/l1/w1_s/AssignAssign1/Critic/target_net/l1/w1_s51/Critic/target_net/l1/w1_s/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
£
 1/Critic/target_net/l1/w1_s/readIdentity1/Critic/target_net/l1/w1_s*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
_output_shapes
:	С*
T0
Љ
;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB"      *
dtype0
ѓ
:1/Critic/target_net/l1/w1_a/Initializer/random_normal/meanConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *    *
dtype0*
_output_shapes
: 
±
<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddevConst*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Я
J1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;1/Critic/target_net/l1/w1_a/Initializer/random_normal/shape*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
seed2Ц*
dtype0*
_output_shapes

:*

seed
£
91/Critic/target_net/l1/w1_a/Initializer/random_normal/mulMulJ1/Critic/target_net/l1/w1_a/Initializer/random_normal/RandomStandardNormal<1/Critic/target_net/l1/w1_a/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a
М
51/Critic/target_net/l1/w1_a/Initializer/random_normalAdd91/Critic/target_net/l1/w1_a/Initializer/random_normal/mul:1/Critic/target_net/l1/w1_a/Initializer/random_normal/mean*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0
њ
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
В
"1/Critic/target_net/l1/w1_a/AssignAssign1/Critic/target_net/l1/w1_a51/Critic/target_net/l1/w1_a/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a*
validate_shape(
Ґ
 1/Critic/target_net/l1/w1_a/readIdentity1/Critic/target_net/l1/w1_a*
_output_shapes

:*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_a
Ѓ
+1/Critic/target_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*,
_class"
 loc:@1/Critic/target_net/l1/b1*
valueB*  А?*
dtype0
ї
1/Critic/target_net/l1/b1
VariableV2*,
_class"
 loc:@1/Critic/target_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
т
 1/Critic/target_net/l1/b1/AssignAssign1/Critic/target_net/l1/b1+1/Critic/target_net/l1/b1/Initializer/Const*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
validate_shape(
Ь
1/Critic/target_net/l1/b1/readIdentity1/Critic/target_net/l1/b1*
T0*,
_class"
 loc:@1/Critic/target_net/l1/b1*
_output_shapes

:
®
1/Critic/target_net/l1/MatMulMatMulS_/s_ 1/Critic/target_net/l1/w1_s/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
¬
1/Critic/target_net/l1/MatMul_1MatMul1/Actor/target_net/a/scaled_a 1/Critic/target_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( 
У
1/Critic/target_net/l1/addAdd1/Critic/target_net/l1/MatMul1/Critic/target_net/l1/MatMul_1*'
_output_shapes
:€€€€€€€€€*
T0
С
1/Critic/target_net/l1/add_1Add1/Critic/target_net/l1/add1/Critic/target_net/l1/b1/read*
T0*'
_output_shapes
:€€€€€€€€€
s
1/Critic/target_net/l1/ReluRelu1/Critic/target_net/l1/add_1*'
_output_shapes
:€€€€€€€€€*
T0
 
B1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shapeConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
љ
A1/Critic/target_net/q/dense/kernel/Initializer/random_normal/meanConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
њ
C1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddevConst*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
і
Q1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalB1/Critic/target_net/q/dense/kernel/Initializer/random_normal/shape*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
seed2®*
dtype0*
_output_shapes

:*

seed*
T0
њ
@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulMulQ1/Critic/target_net/q/dense/kernel/Initializer/random_normal/RandomStandardNormalC1/Critic/target_net/q/dense/kernel/Initializer/random_normal/stddev*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0
®
<1/Critic/target_net/q/dense/kernel/Initializer/random_normalAdd@1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mulA1/Critic/target_net/q/dense/kernel/Initializer/random_normal/mean*
_output_shapes

:*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel
Ќ
"1/Critic/target_net/q/dense/kernel
VariableV2*
shared_name *5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
	container *
shape
:*
dtype0*
_output_shapes

:
Ю
)1/Critic/target_net/q/dense/kernel/AssignAssign"1/Critic/target_net/q/dense/kernel<1/Critic/target_net/q/dense/kernel/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel*
validate_shape(
Ј
'1/Critic/target_net/q/dense/kernel/readIdentity"1/Critic/target_net/q/dense/kernel*
_output_shapes

:*
T0*5
_class+
)'loc:@1/Critic/target_net/q/dense/kernel
і
21/Critic/target_net/q/dense/bias/Initializer/ConstConst*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
valueB*  А?*
dtype0*
_output_shapes
:
Ѕ
 1/Critic/target_net/q/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *3
_class)
'%loc:@1/Critic/target_net/q/dense/bias
К
'1/Critic/target_net/q/dense/bias/AssignAssign 1/Critic/target_net/q/dense/bias21/Critic/target_net/q/dense/bias/Initializer/Const*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(*
_output_shapes
:
≠
%1/Critic/target_net/q/dense/bias/readIdentity 1/Critic/target_net/q/dense/bias*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
_output_shapes
:
 
"1/Critic/target_net/q/dense/MatMulMatMul1/Critic/target_net/l1/Relu'1/Critic/target_net/q/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
¬
#1/Critic/target_net/q/dense/BiasAddBiasAdd"1/Critic/target_net/q/dense/MatMul%1/Critic/target_net/q/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
\
1/target_q/addAddR/r1/target_q/mul*'
_output_shapes
:€€€€€€€€€*
T0
Ц
1/TD_error/SquaredDifferenceSquaredDifference1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
a
1/TD_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
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
 *  А?*
dtype0*
_output_shapes
: 
Н
1/C_train/gradients/FillFill1/C_train/gradients/Shape1/C_train/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
З
61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ƒ
01/C_train/gradients/1/TD_error/Mean_grad/ReshapeReshape1/C_train/gradients/Fill61/C_train/gradients/1/TD_error/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
К
.1/C_train/gradients/1/TD_error/Mean_grad/ShapeShape1/TD_error/SquaredDifference*
out_type0*
_output_shapes
:*
T0
џ
-1/C_train/gradients/1/TD_error/Mean_grad/TileTile01/C_train/gradients/1/TD_error/Mean_grad/Reshape.1/C_train/gradients/1/TD_error/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
М
01/C_train/gradients/1/TD_error/Mean_grad/Shape_1Shape1/TD_error/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
’
-1/C_train/gradients/1/TD_error/Mean_grad/ProdProd01/C_train/gradients/1/TD_error/Mean_grad/Shape_1.1/C_train/gradients/1/TD_error/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
z
01/C_train/gradients/1/TD_error/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
ў
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
Ѕ
01/C_train/gradients/1/TD_error/Mean_grad/MaximumMaximum/1/C_train/gradients/1/TD_error/Mean_grad/Prod_121/C_train/gradients/1/TD_error/Mean_grad/Maximum/y*
_output_shapes
: *
T0
њ
11/C_train/gradients/1/TD_error/Mean_grad/floordivFloorDiv-1/C_train/gradients/1/TD_error/Mean_grad/Prod01/C_train/gradients/1/TD_error/Mean_grad/Maximum*
T0*
_output_shapes
: 
®
-1/C_train/gradients/1/TD_error/Mean_grad/CastCast11/C_train/gradients/1/TD_error/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 
Ћ
01/C_train/gradients/1/TD_error/Mean_grad/truedivRealDiv-1/C_train/gradients/1/TD_error/Mean_grad/Tile-1/C_train/gradients/1/TD_error/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Й
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/ShapeShape1/target_q/add*
T0*
out_type0*
_output_shapes
:
Ю
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1Shape!1/Critic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
Э
K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalarConst1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
в
91/C_train/gradients/1/TD_error/SquaredDifference_grad/MulMul<1/C_train/gradients/1/TD_error/SquaredDifference_grad/scalar01/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
Ў
91/C_train/gradients/1/TD_error/SquaredDifference_grad/subSub1/target_q/add!1/Critic/eval_net/q/dense/BiasAdd1^1/C_train/gradients/1/TD_error/Mean_grad/truediv*
T0*'
_output_shapes
:€€€€€€€€€
к
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/Mul91/C_train/gradients/1/TD_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
К
91/C_train/gradients/1/TD_error/SquaredDifference_grad/SumSum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1K1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
А
=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeReshape91/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
О
;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1Sum;1/C_train/gradients/1/TD_error/SquaredDifference_grad/mul_1M1/C_train/gradients/1/TD_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ж
?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1Reshape;1/C_train/gradients/1/TD_error/SquaredDifference_grad/Sum_1=1/C_train/gradients/1/TD_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
≥
91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegNeg?1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
 
F1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg>^1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape
ж
N1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/TD_error/SquaredDifference_grad/ReshapeG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
а
P1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1Identity91/C_train/gradients/1/TD_error/SquaredDifference_grad/NegG^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg
г
F1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGradP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
п
K1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_depsNoOpG^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradQ^1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1
€
S1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependencyIdentityP1/C_train/gradients/1/TD_error/SquaredDifference_grad/tuple/control_dependency_1L^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@1/C_train/gradients/1/TD_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€*
T0
ч
U1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1IdentityF1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradL^1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ю
@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMulS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency%1/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
Л
B1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/ReluS1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
Џ
J1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_depsNoOpA^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulC^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1
ф
R1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependencyIdentity@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulK^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*S
_classI
GEloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul
с
T1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1IdentityB1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1K^1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1*
_output_shapes

:
и
;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGradR1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency1/Critic/eval_net/l1/Relu*
T0*'
_output_shapes
:€€€€€€€€€
С
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
_output_shapes
:*
T0*
out_type0
М
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
Ч
I1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ж
71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradI1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape71/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
К
91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum;1/C_train/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradK1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ч
=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape91/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
 
D1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape>^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1
ё
L1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeE^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape
џ
N1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1E^1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/group_deps*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1*
_output_shapes

:*
T0
Т
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Ц
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
out_type0*
_output_shapes
:*
T0
С
G1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape91/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
У
51/C_train/gradients/1/Critic/eval_net/l1/add_grad/SumSumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyG1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ф
91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape51/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ч
71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_1SumL1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependencyI1/C_train/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ъ
;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape71/C_train/gradients/1/Critic/eval_net/l1/add_grad/Sum_191/C_train/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ƒ
B1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_depsNoOp:^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape<^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
÷
J1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependencyIdentity91/C_train/gradients/1/Critic/eval_net/l1/add_grad/ReshapeC^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*L
_classB
@>loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
№
L1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1Identity;1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1C^1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1
К
;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulMatMulJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency1/Critic/eval_net/l1/w1_s/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€С*
transpose_b(*
T0
и
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1MatMulS/sJ1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	С*
transpose_b( *
T0
Ћ
E1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_depsNoOp<^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1
б
M1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity;1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMulF^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€С
ё
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1F^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes
:	С
Н
=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMulL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_11/Critic/eval_net/l1/w1_a/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b(
э
?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradientL1/C_train/gradients/1/Critic/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
—
G1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_depsNoOp>^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul@^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1
и
O1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependencyIdentity=1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulH^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul*'
_output_shapes
:€€€€€€€€€
е
Q1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1Identity?1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1H^1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1*
_output_shapes

:
Ф
#1/C_train/beta1_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
•
1/C_train/beta1_power
VariableV2*
shared_name **
_class 
loc:@1/Critic/eval_net/l1/b1*
	container *
shape: *
dtype0*
_output_shapes
: 
Ў
1/C_train/beta1_power/AssignAssign1/C_train/beta1_power#1/C_train/beta1_power/initial_value**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
К
1/C_train/beta1_power/readIdentity1/C_train/beta1_power*
_output_shapes
: *
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
Ф
#1/C_train/beta2_power/initial_valueConst**
_class 
loc:@1/Critic/eval_net/l1/b1*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
•
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
Ў
1/C_train/beta2_power/AssignAssign1/C_train/beta2_power#1/C_train/beta2_power/initial_value*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking(
К
1/C_train/beta2_power/readIdentity1/C_train/beta2_power*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
…
J1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"С     *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
≥
@1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
Ї
:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zerosFillJ1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/shape_as_tensor@1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
ћ
(1/C_train/1/Critic/eval_net/l1/w1_s/Adam
VariableV2*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container *
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name 
†
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_s/Adam:1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
ї
-1/C_train/1/Critic/eval_net/l1/w1_s/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*
_output_shapes
:	С*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s
Ћ
L1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"С     *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
:
µ
B1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
dtype0*
_output_shapes
: 
ј
<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zerosFillL1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/shape_as_tensorB1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
ќ
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1
VariableV2*
shape:	С*
dtype0*
_output_shapes
:	С*
shared_name *,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
	container 
¶
11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
validate_shape(*
_output_shapes
:	С
њ
/1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
_output_shapes
:	С
љ
:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
 
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
Я
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/AssignAssign(1/C_train/1/Critic/eval_net/l1/w1_a/Adam:1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(
Ї
-1/C_train/1/Critic/eval_net/l1/w1_a/Adam/readIdentity(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
_output_shapes

:
њ
<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
dtype0*
_output_shapes

:
ћ
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1
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
•
11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/AssignAssign*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1<1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
validate_shape(*
_output_shapes

:
Њ
/1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/readIdentity*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1*
_output_shapes

:*
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a
є
81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0
∆
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
Ч
-1/C_train/1/Critic/eval_net/l1/b1/Adam/AssignAssign&1/C_train/1/Critic/eval_net/l1/b1/Adam81/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1
і
+1/C_train/1/Critic/eval_net/l1/b1/Adam/readIdentity&1/C_train/1/Critic/eval_net/l1/b1/Adam**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:*
T0
ї
:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zerosConst*
valueB*    **
_class 
loc:@1/Critic/eval_net/l1/b1*
dtype0*
_output_shapes

:
»
(1/C_train/1/Critic/eval_net/l1/b1/Adam_1
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
Э
/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/AssignAssign(1/C_train/1/Critic/eval_net/l1/b1/Adam_1:1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
Є
-1/C_train/1/Critic/eval_net/l1/b1/Adam_1/readIdentity(1/C_train/1/Critic/eval_net/l1/b1/Adam_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes

:
Ћ
A1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
Ў
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
ї
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/kernel/AdamA1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(*
_output_shapes

:
ѕ
41/C_train/1/Critic/eval_net/q/dense/kernel/Adam/readIdentity/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
Ќ
C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
dtype0*
_output_shapes

:
Џ
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
Ѕ
81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/AssignAssign11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
validate_shape(
”
61/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/readIdentity11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1*
_output_shapes

:*
T0*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel
њ
?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
ћ
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
ѓ
41/C_train/1/Critic/eval_net/q/dense/bias/Adam/AssignAssign-1/C_train/1/Critic/eval_net/q/dense/bias/Adam?1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(*
_output_shapes
:
≈
21/C_train/1/Critic/eval_net/q/dense/bias/Adam/readIdentity-1/C_train/1/Critic/eval_net/q/dense/bias/Adam*
_output_shapes
:*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
Ѕ
A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
dtype0*
_output_shapes
:
ќ
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
µ
61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/AssignAssign/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1A1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
validate_shape(
…
41/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/readIdentity/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1*
_output_shapes
:*
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias
a
1/C_train/Adam/learning_rateConst*
valueB
 *Ј—8*
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
 *wЊ?*
dtype0*
_output_shapes
: 
[
1/C_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
°
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_s(1/C_train/1/Critic/eval_net/l1/w1_s/Adam*1/C_train/1/Critic/eval_net/l1/w1_s/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonO1/C_train/gradients/1/Critic/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	С*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_s*
use_nesterov( 
Ґ
91/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/w1_a(1/C_train/1/Critic/eval_net/l1/w1_a/Adam*1/C_train/1/Critic/eval_net/l1/w1_a/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonQ1/C_train/gradients/1/Critic/eval_net/l1/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@1/Critic/eval_net/l1/w1_a*
use_nesterov( *
_output_shapes

:
Х
71/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam	ApplyAdam1/Critic/eval_net/l1/b1&1/C_train/1/Critic/eval_net/l1/b1/Adam(1/C_train/1/Critic/eval_net/l1/b1/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonN1/C_train/gradients/1/Critic/eval_net/l1/add_1_grad/tuple/control_dependency_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
use_nesterov( *
_output_shapes

:*
use_locking( 
»
@1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam	ApplyAdam 1/Critic/eval_net/q/dense/kernel/1/C_train/1/Critic/eval_net/q/dense/kernel/Adam11/C_train/1/Critic/eval_net/q/dense/kernel/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonT1/C_train/gradients/1/Critic/eval_net/q/dense/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@1/Critic/eval_net/q/dense/kernel*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
ї
>1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdam	ApplyAdam1/Critic/eval_net/q/dense/bias-1/C_train/1/Critic/eval_net/q/dense/bias/Adam/1/C_train/1/Critic/eval_net/q/dense/bias/Adam_11/C_train/beta1_power/read1/C_train/beta2_power/read1/C_train/Adam/learning_rate1/C_train/Adam/beta11/C_train/Adam/beta21/C_train/Adam/epsilonU1/C_train/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*1
_class'
%#loc:@1/Critic/eval_net/q/dense/bias*
use_nesterov( *
_output_shapes
:
ќ
1/C_train/Adam/mulMul1/C_train/beta1_power/read1/C_train/Adam/beta18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
ј
1/C_train/Adam/AssignAssign1/C_train/beta1_power1/C_train/Adam/mul**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
–
1/C_train/Adam/mul_1Mul1/C_train/beta2_power/read1/C_train/Adam/beta28^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
_output_shapes
: 
ƒ
1/C_train/Adam/Assign_1Assign1/C_train/beta2_power1/C_train/Adam/mul_1*
T0**
_class 
loc:@1/Critic/eval_net/l1/b1*
validate_shape(*
_output_shapes
: *
use_locking( 
ю
1/C_train/AdamNoOp^1/C_train/Adam/Assign^1/C_train/Adam/Assign_18^1/C_train/Adam/update_1/Critic/eval_net/l1/b1/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_a/ApplyAdam:^1/C_train/Adam/update_1/Critic/eval_net/l1/w1_s/ApplyAdam?^1/C_train/Adam/update_1/Critic/eval_net/q/dense/bias/ApplyAdamA^1/C_train/Adam/update_1/Critic/eval_net/q/dense/kernel/ApplyAdam
y
1/a_grad/gradients/ShapeShape!1/Critic/eval_net/q/dense/BiasAdd*
_output_shapes
:*
T0*
out_type0
a
1/a_grad/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ы
1/a_grad/gradients/FillFill1/a_grad/gradients/Shape1/a_grad/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
©
E1/a_grad/gradients/1/Critic/eval_net/q/dense/BiasAdd_grad/BiasAddGradBiasAddGrad1/a_grad/gradients/Fill*
T0*
data_formatNHWC*
_output_shapes
:
б
?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMulMatMul1/a_grad/gradients/Fill%1/Critic/eval_net/q/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
ќ
A1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul_1MatMul1/Critic/eval_net/l1/Relu1/a_grad/gradients/Fill*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
‘
:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradReluGrad?1/a_grad/gradients/1/Critic/eval_net/q/dense/MatMul_grad/MatMul1/Critic/eval_net/l1/Relu*'
_output_shapes
:€€€€€€€€€*
T0
Р
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ShapeShape1/Critic/eval_net/l1/add*
out_type0*
_output_shapes
:*
T0
Л
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"      
Ф
H1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Г
61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradH1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ч
:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeReshape61/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
З
81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/Relu_grad/ReluGradJ1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ф
<1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Reshape_1Reshape81/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Sum_1:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
С
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ShapeShape1/Critic/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
Х
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1Shape1/Critic/eval_net/l1/MatMul_1*
T0*
out_type0*
_output_shapes
:
О
F1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/SumSum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeF1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
с
81/a_grad/gradients/1/Critic/eval_net/l1/add_grad/ReshapeReshape41/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Г
61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_1Sum:1/a_grad/gradients/1/Critic/eval_net/l1/add_1_grad/ReshapeH1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ч
:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1Reshape61/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Sum_181/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ъ
<1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMulMatMul:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_11/Critic/eval_net/l1/w1_a/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
к
>1/a_grad/gradients/1/Critic/eval_net/l1/MatMul_1_grad/MatMul_1MatMul1/Critic/StopGradient:1/a_grad/gradients/1/Critic/eval_net/l1/add_grad/Reshape_1*
T0*
transpose_a(*
_output_shapes

:*
transpose_b( 
N
	1/mul_8/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
e
1/mul_8Mul	1/mul_8/x 1/Critic/target_net/l1/w1_s/read*
_output_shapes
:	С*
T0
N
	1/mul_9/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
c
1/mul_9Mul	1/mul_9/x1/Critic/eval_net/l1/w1_s/read*
_output_shapes
:	С*
T0
J
1/add_4Add1/mul_81/mul_9*
T0*
_output_shapes
:	С
љ

1/Assign_4Assign1/Critic/target_net/l1/w1_s1/add_4*
_output_shapes
:	С*
use_locking(*
T0*.
_class$
" loc:@1/Critic/target_net/l1/w1_s*
validate_shape(
O

1/mul_10/xConst*
valueB
 *§p}?*
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
„#<*
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
1/add_5Add1/mul_101/mul_11*
T0*
_output_shapes

:
Љ

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
 *§p}?*
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

1/mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
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
Є

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
 *§p}?*
dtype0*
_output_shapes
: 
m
1/mul_14Mul
1/mul_14/x'1/Critic/target_net/q/dense/kernel/read*
_output_shapes

:*
T0
O

1/mul_15/xConst*
valueB
 *
„#<*
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
1/add_7Add1/mul_141/mul_15*
T0*
_output_shapes

:
 

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
 *§p}?*
dtype0
g
1/mul_16Mul
1/mul_16/x%1/Critic/target_net/q/dense/bias/read*
_output_shapes
:*
T0
O

1/mul_17/xConst*
valueB
 *
„#<*
dtype0*
_output_shapes
: 
e
1/mul_17Mul
1/mul_17/x#1/Critic/eval_net/q/dense/bias/read*
_output_shapes
:*
T0
G
1/add_8Add1/mul_161/mul_17*
T0*
_output_shapes
:
¬

1/Assign_8Assign 1/Critic/target_net/q/dense/bias1/add_8*
_output_shapes
:*
use_locking(*
T0*3
_class)
'%loc:@1/Critic/target_net/q/dense/bias*
validate_shape(
y
1/policy_grads/gradients/ShapeShape1/Actor/eval_net/a/scaled_a*
T0*
out_type0*
_output_shapes
:
g
"1/policy_grads/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
≠
1/policy_grads/gradients/FillFill1/policy_grads/gradients/Shape"1/policy_grads/gradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
Ы
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeShape1/Actor/eval_net/a/a/Sigmoid*
_output_shapes
:*
T0*
out_type0
Д
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
©
O1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgsBroadcastGradientArgs?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ShapeA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulMul1/policy_grads/gradients/Fill1/Actor/eval_net/a/scaled_a/y*'
_output_shapes
:€€€€€€€€€*
T0
Ф
=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/SumSum=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/MulO1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
М
A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/ReshapeReshape=1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
µ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Mul1/Actor/eval_net/a/a/Sigmoid1/policy_grads/gradients/Fill*
T0*'
_output_shapes
:€€€€€€€€€
Ъ
?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1Sum?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Mul_1Q1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
C1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape_1Reshape?1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Sum_1A1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
и
F1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGradSigmoidGrad1/Actor/eval_net/a/a/SigmoidA1/policy_grads/gradients/1/Actor/eval_net/a/scaled_a_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
ў
F1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGradBiasAddGradF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
М
@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMulMatMulF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad 1/Actor/eval_net/a/a/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€d*
transpose_b(*
T0
э
B1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1MatMul1/Actor/eval_net/l1/TanhF1/policy_grads/gradients/1/Actor/eval_net/a/a/Sigmoid_grad/SigmoidGrad*
T0*
transpose_a(*
_output_shapes

:d*
transpose_b( 
ў
?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGradTanhGrad1/Actor/eval_net/l1/Tanh@1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€d*
T0
—
E1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGradBiasAddGrad?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Д
?1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMulMatMul?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad1/Actor/eval_net/l1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€С*
transpose_b(
б
A1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1MatMulS/s?1/policy_grads/gradients/1/Actor/eval_net/l1/Tanh_grad/TanhGrad*
T0*
transpose_a(*
_output_shapes
:	Сd*
transpose_b( 
Ц
#1/A_train/beta1_power/initial_valueConst*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
І
1/A_train/beta1_power
VariableV2*
shared_name *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
Џ
1/A_train/beta1_power/AssignAssign1/A_train/beta1_power#1/A_train/beta1_power/initial_value*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
М
1/A_train/beta1_power/readIdentity1/A_train/beta1_power*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
Ц
#1/A_train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
valueB
 *wЊ?
І
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
Џ
1/A_train/beta2_power/AssignAssign1/A_train/beta2_power#1/A_train/beta2_power/initial_value*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking(
М
1/A_train/beta2_power/readIdentity1/A_train/beta2_power*
_output_shapes
: *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
Ћ
K1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"С  d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
µ
A1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
Њ
;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zerosFillK1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/shape_as_tensorA1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
ќ
)1/A_train/1/Actor/eval_net/l1/kernel/Adam
VariableV2*
shape:	Сd*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container 
§
01/A_train/1/Actor/eval_net/l1/kernel/Adam/AssignAssign)1/A_train/1/Actor/eval_net/l1/kernel/Adam;1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd
Њ
.1/A_train/1/Actor/eval_net/l1/kernel/Adam/readIdentity)1/A_train/1/Actor/eval_net/l1/kernel/Adam*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0
Ќ
M1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"С  d   *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
:
Ј
C1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
dtype0*
_output_shapes
: 
ƒ
=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zerosFillM1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/shape_as_tensorC1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd
–
+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	Сd*
shared_name *-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
	container *
shape:	Сd
™
21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/AssignAssign+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1=1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
validate_shape(*
_output_shapes
:	Сd*
use_locking(*
T0
¬
01/A_train/1/Actor/eval_net/l1/kernel/Adam_1/readIdentity+1/A_train/1/Actor/eval_net/l1/kernel/Adam_1*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
_output_shapes
:	Сd*
T0
≥
91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zerosConst*
valueBd*    *+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
ј
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
Ч
.1/A_train/1/Actor/eval_net/l1/bias/Adam/AssignAssign'1/A_train/1/Actor/eval_net/l1/bias/Adam91/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
validate_shape(*
_output_shapes
:d
≥
,1/A_train/1/Actor/eval_net/l1/bias/Adam/readIdentity'1/A_train/1/Actor/eval_net/l1/bias/Adam*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
µ
;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zerosConst*
valueBd*    *+
_class!
loc:@1/Actor/eval_net/l1/bias*
dtype0*
_output_shapes
:d
¬
)1/A_train/1/Actor/eval_net/l1/bias/Adam_1
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *+
_class!
loc:@1/Actor/eval_net/l1/bias*
	container 
Э
01/A_train/1/Actor/eval_net/l1/bias/Adam_1/AssignAssign)1/A_train/1/Actor/eval_net/l1/bias/Adam_1;1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias
Ј
.1/A_train/1/Actor/eval_net/l1/bias/Adam_1/readIdentity)1/A_train/1/Actor/eval_net/l1/bias/Adam_1*
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
_output_shapes
:d
Ѕ
<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
ќ
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
І
11/A_train/1/Actor/eval_net/a/a/kernel/Adam/AssignAssign*1/A_train/1/Actor/eval_net/a/a/kernel/Adam<1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
ј
/1/A_train/1/Actor/eval_net/a/a/kernel/Adam/readIdentity*1/A_train/1/Actor/eval_net/a/a/kernel/Adam*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
√
>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zerosConst*
valueBd*    *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
dtype0*
_output_shapes

:d
–
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:d*
shared_name *.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
	container *
shape
:d
≠
31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/AssignAssign,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1>1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
validate_shape(*
_output_shapes

:d
ƒ
11/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/readIdentity,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1*
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
_output_shapes

:d
µ
:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
¬
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
Ы
/1/A_train/1/Actor/eval_net/a/a/bias/Adam/AssignAssign(1/A_train/1/Actor/eval_net/a/a/bias/Adam:1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:*
use_locking(
ґ
-1/A_train/1/Actor/eval_net/a/a/bias/Adam/readIdentity(1/A_train/1/Actor/eval_net/a/a/bias/Adam*
_output_shapes
:*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
Ј
<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
dtype0*
_output_shapes
:
ƒ
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
°
11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/AssignAssign*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1<1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
:
Ї
/1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/readIdentity*1/A_train/1/Actor/eval_net/a/a/bias/Adam_1*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
:
a
1/A_train/Adam/learning_rateConst*
valueB
 *Ј—Є*
dtype0*
_output_shapes
: 
Y
1/A_train/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Y
1/A_train/Adam/beta2Const*
_output_shapes
: *
valueB
 *wЊ?*
dtype0
[
1/A_train/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Ш
:1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/kernel)1/A_train/1/Actor/eval_net/l1/kernel/Adam+1/A_train/1/Actor/eval_net/l1/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonA1/policy_grads/gradients/1/Actor/eval_net/l1/MatMul_grad/MatMul_1*-
_class#
!loc:@1/Actor/eval_net/l1/kernel*
use_nesterov( *
_output_shapes
:	Сd*
use_locking( *
T0
Н
81/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/l1/bias'1/A_train/1/Actor/eval_net/l1/bias/Adam)1/A_train/1/Actor/eval_net/l1/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonE1/policy_grads/gradients/1/Actor/eval_net/l1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@1/Actor/eval_net/l1/bias*
use_nesterov( *
_output_shapes
:d
Э
;1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/kernel*1/A_train/1/Actor/eval_net/a/a/kernel/Adam,1/A_train/1/Actor/eval_net/a/a/kernel/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonB1/policy_grads/gradients/1/Actor/eval_net/a/a/MatMul_grad/MatMul_1*
use_locking( *
T0*.
_class$
" loc:@1/Actor/eval_net/a/a/kernel*
use_nesterov( *
_output_shapes

:d
У
91/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam	ApplyAdam1/Actor/eval_net/a/a/bias(1/A_train/1/Actor/eval_net/a/a/bias/Adam*1/A_train/1/Actor/eval_net/a/a/bias/Adam_11/A_train/beta1_power/read1/A_train/beta2_power/read1/A_train/Adam/learning_rate1/A_train/Adam/beta11/A_train/Adam/beta21/A_train/Adam/epsilonF1/policy_grads/gradients/1/Actor/eval_net/a/a/BiasAdd_grad/BiasAddGrad*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
М
1/A_train/Adam/mulMul1/A_train/beta1_power/read1/A_train/Adam/beta1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias
¬
1/A_train/Adam/AssignAssign1/A_train/beta1_power1/A_train/Adam/mul*
use_locking( *
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: 
О
1/A_train/Adam/mul_1Mul1/A_train/beta2_power/read1/A_train/Adam/beta2:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam*
T0*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
_output_shapes
: 
∆
1/A_train/Adam/Assign_1Assign1/A_train/beta2_power1/A_train/Adam/mul_1*,
_class"
 loc:@1/Actor/eval_net/a/a/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Ї
1/A_train/AdamNoOp^1/A_train/Adam/Assign^1/A_train/Adam/Assign_1:^1/A_train/Adam/update_1/Actor/eval_net/a/a/bias/ApplyAdam<^1/A_train/Adam/update_1/Actor/eval_net/a/a/kernel/ApplyAdam9^1/A_train/Adam/update_1/Actor/eval_net/l1/bias/ApplyAdam;^1/A_train/Adam/update_1/Actor/eval_net/l1/kernel/ApplyAdam
÷
initNoOp0^1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign2^1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign2^1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign4^1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign/^1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign1^1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign1^1/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign3^1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign^1/A_train/beta1_power/Assign^1/A_train/beta2_power/Assign!^1/Actor/eval_net/a/a/bias/Assign#^1/Actor/eval_net/a/a/kernel/Assign ^1/Actor/eval_net/l1/bias/Assign"^1/Actor/eval_net/l1/kernel/Assign#^1/Actor/target_net/a/a/bias/Assign%^1/Actor/target_net/a/a/kernel/Assign"^1/Actor/target_net/l1/bias/Assign$^1/Actor/target_net/l1/kernel/Assign.^1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign0^1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign0^1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign2^1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign5^1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign7^1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign7^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign9^1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign^1/C_train/beta1_power/Assign^1/C_train/beta2_power/Assign^1/Critic/eval_net/l1/b1/Assign!^1/Critic/eval_net/l1/w1_a/Assign!^1/Critic/eval_net/l1/w1_s/Assign&^1/Critic/eval_net/q/dense/bias/Assign(^1/Critic/eval_net/q/dense/kernel/Assign!^1/Critic/target_net/l1/b1/Assign#^1/Critic/target_net/l1/w1_a/Assign#^1/Critic/target_net/l1/w1_s/Assign(^1/Critic/target_net/q/dense/bias/Assign*^1/Critic/target_net/q/dense/kernel/Assign"&"Я9
	variablesС9О9
Ю
1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08
О
1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
Ґ
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
Т
1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08
§
1/Actor/target_net/l1/kernel:0#1/Actor/target_net/l1/kernel/Assign#1/Actor/target_net/l1/kernel/read:0281/Actor/target_net/l1/kernel/Initializer/random_normal:0
Ф
1/Actor/target_net/l1/bias:0!1/Actor/target_net/l1/bias/Assign!1/Actor/target_net/l1/bias/read:02.1/Actor/target_net/l1/bias/Initializer/Const:0
®
1/Actor/target_net/a/a/kernel:0$1/Actor/target_net/a/a/kernel/Assign$1/Actor/target_net/a/a/kernel/read:0291/Actor/target_net/a/a/kernel/Initializer/random_normal:0
Ш
1/Actor/target_net/a/a/bias:0"1/Actor/target_net/a/a/bias/Assign"1/Actor/target_net/a/a/bias/read:02/1/Actor/target_net/a/a/bias/Initializer/Const:0
Ъ
1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Ъ
1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
К
1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
ґ
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
¶
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08
†
1/Critic/target_net/l1/w1_s:0"1/Critic/target_net/l1/w1_s/Assign"1/Critic/target_net/l1/w1_s/read:0271/Critic/target_net/l1/w1_s/Initializer/random_normal:0
†
1/Critic/target_net/l1/w1_a:0"1/Critic/target_net/l1/w1_a/Assign"1/Critic/target_net/l1/w1_a/read:0271/Critic/target_net/l1/w1_a/Initializer/random_normal:0
Р
1/Critic/target_net/l1/b1:0 1/Critic/target_net/l1/b1/Assign 1/Critic/target_net/l1/b1/read:02-1/Critic/target_net/l1/b1/Initializer/Const:0
Љ
$1/Critic/target_net/q/dense/kernel:0)1/Critic/target_net/q/dense/kernel/Assign)1/Critic/target_net/q/dense/kernel/read:02>1/Critic/target_net/q/dense/kernel/Initializer/random_normal:0
ђ
"1/Critic/target_net/q/dense/bias:0'1/Critic/target_net/q/dense/bias/Assign'1/Critic/target_net/q/dense/bias/read:0241/Critic/target_net/q/dense/bias/Initializer/Const:0
|
1/C_train/beta1_power:01/C_train/beta1_power/Assign1/C_train/beta1_power/read:02%1/C_train/beta1_power/initial_value:0
|
1/C_train/beta2_power:01/C_train/beta2_power/Assign1/C_train/beta2_power/read:02%1/C_train/beta2_power/initial_value:0
ћ
*1/C_train/1/Critic/eval_net/l1/w1_s/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_s/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_s/Adam/Initializer/zeros:0
‘
,1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_s/Adam_1/Initializer/zeros:0
ћ
*1/C_train/1/Critic/eval_net/l1/w1_a/Adam:0/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Assign/1/C_train/1/Critic/eval_net/l1/w1_a/Adam/read:02<1/C_train/1/Critic/eval_net/l1/w1_a/Adam/Initializer/zeros:0
‘
,1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1:011/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Assign11/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/read:02>1/C_train/1/Critic/eval_net/l1/w1_a/Adam_1/Initializer/zeros:0
ƒ
(1/C_train/1/Critic/eval_net/l1/b1/Adam:0-1/C_train/1/Critic/eval_net/l1/b1/Adam/Assign-1/C_train/1/Critic/eval_net/l1/b1/Adam/read:02:1/C_train/1/Critic/eval_net/l1/b1/Adam/Initializer/zeros:0
ћ
*1/C_train/1/Critic/eval_net/l1/b1/Adam_1:0/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Assign/1/C_train/1/Critic/eval_net/l1/b1/Adam_1/read:02<1/C_train/1/Critic/eval_net/l1/b1/Adam_1/Initializer/zeros:0
и
11/C_train/1/Critic/eval_net/q/dense/kernel/Adam:061/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Assign61/C_train/1/Critic/eval_net/q/dense/kernel/Adam/read:02C1/C_train/1/Critic/eval_net/q/dense/kernel/Adam/Initializer/zeros:0
р
31/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1:081/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Assign81/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/read:02E1/C_train/1/Critic/eval_net/q/dense/kernel/Adam_1/Initializer/zeros:0
а
/1/C_train/1/Critic/eval_net/q/dense/bias/Adam:041/C_train/1/Critic/eval_net/q/dense/bias/Adam/Assign41/C_train/1/Critic/eval_net/q/dense/bias/Adam/read:02A1/C_train/1/Critic/eval_net/q/dense/bias/Adam/Initializer/zeros:0
и
11/C_train/1/Critic/eval_net/q/dense/bias/Adam_1:061/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Assign61/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/read:02C1/C_train/1/Critic/eval_net/q/dense/bias/Adam_1/Initializer/zeros:0
|
1/A_train/beta1_power:01/A_train/beta1_power/Assign1/A_train/beta1_power/read:02%1/A_train/beta1_power/initial_value:0
|
1/A_train/beta2_power:01/A_train/beta2_power/Assign1/A_train/beta2_power/read:02%1/A_train/beta2_power/initial_value:0
–
+1/A_train/1/Actor/eval_net/l1/kernel/Adam:001/A_train/1/Actor/eval_net/l1/kernel/Adam/Assign01/A_train/1/Actor/eval_net/l1/kernel/Adam/read:02=1/A_train/1/Actor/eval_net/l1/kernel/Adam/Initializer/zeros:0
Ў
-1/A_train/1/Actor/eval_net/l1/kernel/Adam_1:021/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Assign21/A_train/1/Actor/eval_net/l1/kernel/Adam_1/read:02?1/A_train/1/Actor/eval_net/l1/kernel/Adam_1/Initializer/zeros:0
»
)1/A_train/1/Actor/eval_net/l1/bias/Adam:0.1/A_train/1/Actor/eval_net/l1/bias/Adam/Assign.1/A_train/1/Actor/eval_net/l1/bias/Adam/read:02;1/A_train/1/Actor/eval_net/l1/bias/Adam/Initializer/zeros:0
–
+1/A_train/1/Actor/eval_net/l1/bias/Adam_1:001/A_train/1/Actor/eval_net/l1/bias/Adam_1/Assign01/A_train/1/Actor/eval_net/l1/bias/Adam_1/read:02=1/A_train/1/Actor/eval_net/l1/bias/Adam_1/Initializer/zeros:0
‘
,1/A_train/1/Actor/eval_net/a/a/kernel/Adam:011/A_train/1/Actor/eval_net/a/a/kernel/Adam/Assign11/A_train/1/Actor/eval_net/a/a/kernel/Adam/read:02>1/A_train/1/Actor/eval_net/a/a/kernel/Adam/Initializer/zeros:0
№
.1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1:031/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Assign31/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/read:02@1/A_train/1/Actor/eval_net/a/a/kernel/Adam_1/Initializer/zeros:0
ћ
*1/A_train/1/Actor/eval_net/a/a/bias/Adam:0/1/A_train/1/Actor/eval_net/a/a/bias/Adam/Assign/1/A_train/1/Actor/eval_net/a/a/bias/Adam/read:02<1/A_train/1/Actor/eval_net/a/a/bias/Adam/Initializer/zeros:0
‘
,1/A_train/1/Actor/eval_net/a/a/bias/Adam_1:011/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Assign11/A_train/1/Actor/eval_net/a/a/bias/Adam_1/read:02>1/A_train/1/Actor/eval_net/a/a/bias/Adam_1/Initializer/zeros:0"∞
trainable_variablesШХ
Ю
1/Actor/eval_net/l1/kernel:0!1/Actor/eval_net/l1/kernel/Assign!1/Actor/eval_net/l1/kernel/read:0261/Actor/eval_net/l1/kernel/Initializer/random_normal:08
О
1/Actor/eval_net/l1/bias:01/Actor/eval_net/l1/bias/Assign1/Actor/eval_net/l1/bias/read:02,1/Actor/eval_net/l1/bias/Initializer/Const:08
Ґ
1/Actor/eval_net/a/a/kernel:0"1/Actor/eval_net/a/a/kernel/Assign"1/Actor/eval_net/a/a/kernel/read:0271/Actor/eval_net/a/a/kernel/Initializer/random_normal:08
Т
1/Actor/eval_net/a/a/bias:0 1/Actor/eval_net/a/a/bias/Assign 1/Actor/eval_net/a/a/bias/read:02-1/Actor/eval_net/a/a/bias/Initializer/Const:08
Ъ
1/Critic/eval_net/l1/w1_s:0 1/Critic/eval_net/l1/w1_s/Assign 1/Critic/eval_net/l1/w1_s/read:0251/Critic/eval_net/l1/w1_s/Initializer/random_normal:08
Ъ
1/Critic/eval_net/l1/w1_a:0 1/Critic/eval_net/l1/w1_a/Assign 1/Critic/eval_net/l1/w1_a/read:0251/Critic/eval_net/l1/w1_a/Initializer/random_normal:08
К
1/Critic/eval_net/l1/b1:01/Critic/eval_net/l1/b1/Assign1/Critic/eval_net/l1/b1/read:02+1/Critic/eval_net/l1/b1/Initializer/Const:08
ґ
"1/Critic/eval_net/q/dense/kernel:0'1/Critic/eval_net/q/dense/kernel/Assign'1/Critic/eval_net/q/dense/kernel/read:02<1/Critic/eval_net/q/dense/kernel/Initializer/random_normal:08
¶
 1/Critic/eval_net/q/dense/bias:0%1/Critic/eval_net/q/dense/bias/Assign%1/Critic/eval_net/q/dense/bias/read:0221/Critic/eval_net/q/dense/bias/Initializer/Const:08".
train_op"
 
1/C_train/Adam
1/A_train/AdamІгр