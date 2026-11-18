이전 버전은 위에서 수직으로 누를 때만 맞았는데, 이번에는 면마다 패드를 달아서 비스듬히 누르기, 양옆에서 잡기, 들어 올리기까지 됩니다.

힘 (FEM 결과)

위에서 누를 때, 깊이 1 / 2 / 3 mm

    soft    0.36 / 1.00 / 1.81 N
    medium  0.72 / 2.00 / 3.62 N
    hard    1.25 / 3.50 / 6.34 N

양옆에서 잡을 때, 닫힘 1 / 2 / 3 mm, 손가락 한쪽당

    soft    0.14 / 0.33 / 0.53 N
    medium  0.29 / 0.66 / 1.06 N
    hard    0.50 / 1.15 / 1.86 N

위에서 누르는 쪽은 반지름 10 mm 구형 프로브 기준이고, 잡는 쪽은 실제 SO-101 그리퍼 형상(공식 STL)으로 큐브를 깊게 잡았을 때 기준입니다. 닫힘은 그리퍼가 큐브에 처음 닿은 뒤에 더 닫힌 거리입니다.

구조

큐브는 가운데 박스 하나에 패드 3개(윗면, -x면, +x면)가 붙어 있습니다. 패드는 자기 면의 법선 방향으로만 움직이고, 눌린 깊이에 맞춰 FEM 곡선대로 힘을 냅니다. 그래서 누르는 물체의 질량이나 로봇 자세가 바뀌어도 힘은 같습니다.

큐브 본체에는 free joint가 있어서 잡고 들 수 있습니다.

고정 손가락 쪽이 -x면, 움직이는 손가락 쪽이 +x면입니다. 반대로 잡으실 거면 make_cubes.py --fixed-jaw pos 로 다시 만들어 주세요.

실행

    pip install mujoco numpy
    python demo_grip.py medium
    python demo_grip.py hard --close 2.5
    python demo_grip.py medium --no-viewer
    python validate_faces.py

demo_grip.py는 잡고, 들고, 내려놓는 데모입니다. grip_demo.mp4가 그 영상입니다. 데모의 그리퍼는 그냥 박스인데, 실제 그리퍼 모양은 옆면 패드의 힘 곡선에 들어가 있습니다.

장면에 넣기

XML에는 아래 두 줄이 필요합니다. option 줄이 없으면 들었을 때 큐브가 조금씩 미끄러집니다.

    <option cone="elliptic" impratio="10"/>
    <include file="cubes/cube_medium.xml"/>

파이썬 쪽은 이전과 같습니다.

    from cube_force import CubeForces
    cubes = CubeForces(model, "cubes/cube_params.json")
    cubes.apply(data)      (매 스텝, mj_step 전에)
    mujoco.mj_step(model, data)

값을 읽을 때는

    cubes.force(data)["medium"]         윗면 (깊이 mm, 힘 N). 이전과 같습니다
    cubes.faces(data)["medium"]         면별 (깊이, 힘). top, xneg, xpos
    cubes.force_world(data)["medium"]   면별 힘 벡터 3개 성분, 월드 좌표
    cubes.grip(data)["medium"]          (닫힘 mm, 손가락 한쪽당 힘 N)

cube_params.json 형식이 바뀌어서 cubes 폴더는 통째로 바꿔 주셔야 합니다.

큐브 위치나 질량을 바꾸려면

    python make_cubes.py --pos 0.20 0 0 --mass 0.015
    python make_cubes.py --fixed          (free joint 없이 고정)
    python make_cubes.py --solref 0.01    (timestep 2 ms일 때)

확인한 것

validate_faces.py를 돌린 결과가 validation_output.txt에 있습니다.

위에서 누를 때 프로브 질량을 0.02, 0.1, 0.5 kg으로 바꿔도 접촉력이 FEM 곡선과 같았습니다.

0, 15, 30, 45도로 비스듬히 눌렀을 때 법선 방향 힘이 FEM 값과 같았습니다.

잡았을 때 양쪽 힘이 같고, 그리퍼 FEM 곡선과 맞았습니다.

20 mm 들어서 0.9초 유지하는 동안 높이가 0.1 mm도 안 변했습니다.

timestep 1 ms에서는 문제가 없었습니다. 2 ms에서는 기본 설정으로 hard 큐브를 들다가 떨어뜨려서, --solref 0.01 로 만들어야 합니다.

아직 안 되는 것

옆면 곡선은 30 mm 큐브를 깊게 잡았을 때(움직이는 손가락이 27도쯤 열림) 값입니다. 손가락 끝으로 잡으면 힘이 더 크게 나올 텐데, 잡는 위치가 정해지면 다시 계산하겠습니다.

그리퍼 안쪽 폼 패드는 아직 계산에 없습니다.

큐브 질량 15 g은 임시 값입니다.

윗면 패드는 x 방향으로 가운데 22 mm만 차지합니다. 양쪽 끝 4 mm는 옆면 패드라서, 위에서는 가운데를 눌러 주세요.

비스듬히 누르고 멈춰 있으면 접선 방향 힘이 0으로 나옵니다. 마찰이 없을 때와 같은 결과이고, 실제로는 마찰 때문에 힘이 더 클 수 있습니다.

옆면을 프로브로 누르거나 윗면을 그리퍼로 잡는 경우는 맞지 않습니다.

닫힘은 서보 목표보다 0.05에서 0.6 mm 정도 작게 나옵니다. 힘은 실제 패드 깊이로 계산합니다.
