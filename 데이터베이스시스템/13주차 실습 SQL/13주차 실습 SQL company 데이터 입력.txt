USE company;

-- 부서테이블 데이터 입력
-- 외래키(mgrssn)은 나중에 입력
INSERT INTO department (dnumber,dname,mgrstartdate) VALUES (1, '영업부', '2015/04/18');
INSERT INTO department (dnumber,dname,mgrstartdate) VALUES (4, '총무부', '2017/03/21');
INSERT INTO department (dnumber,dname,mgrstartdate) VALUES (5, '연구소', '2018/09/11');
SELECT * FROM department;

-- 사원테이블 데이터 입력
-- 외래키에 유의해서 입력(직속상사가 먼저 입력되어야 함)
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('강명석','333445555','서울 송파구 풍납동',5000,'M','1965/12/08',5,NULL);
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('김유헌','987987987','서울 강남구 압구정동',4800,'M','1968/12/12',4,NULL);
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('박현식','444889999','서울 강동구 암사동',4000,'M','1975/10/09',1,NULL);
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('한명숙','666884444','경기 용인시 풍덕천동',4300,'F','1972/07/05',5,'333445555');
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('김창헌','123456789','서울 광진구 구의동',4000,'M','1975/01/09',5,'666884444');
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('남석훈','999887777','경기 구리시 인창동',4000,'M','1978/06/19',4,'987987987');
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('전명희','987654321','서울 종로구 안국동',3500,'F','1981/06/30',4,'987987987');
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('이혜진','453453453','경기 용인시 동천동',4000,'F','1975/09/12',5,'666884444');
INSERT INTO employee (ename,ssn,address,salary,sex,bdate,dno,superssn)
VALUES ('이지은','888665555','서울 강동구 고덕동',3800,'F','1977/12/11',1,'444889999');
SELECT * FROM employee;

-- 부서 테이블 관리자 입력(수정)
UPDATE department SET mgrssn='333445555' WHERE dnumber=1;
UPDATE department SET mgrssn='987987987' WHERE dnumber=4;
UPDATE department SET mgrssn='444889999' WHERE dnumber=5;
SELECT * FROM department;

-- 부서위치 테이블 데이터입력
INSERT INTO dept_locations(dnumber,dlocations) VALUES (1,'서울');
INSERT INTO dept_locations(dnumber,dlocations) VALUES (1,'부산');
INSERT INTO dept_locations(dnumber,dlocations) VALUES (1,'대구');
INSERT INTO dept_locations(dnumber,dlocations) VALUES (4,'서울');
INSERT INTO dept_locations(dnumber,dlocations) VALUES (4,'인천');
INSERT INTO dept_locations(dnumber,dlocations) VALUES (5,'대전');
SELECT * FROM dept_locations;

-- 프로젝트 테이블 데이터 입력
INSERT INTO project(pnumber,pname,plocation,dnum)
VALUES (10, '차세대 정보 시스템 도입', '서울', 5);
INSERT INTO project(pnumber,pname,plocation,dnum)
VALUES (20, '효과적 조직 개편안 연구', '서울', 4);
INSERT INTO project(pnumber,pname,plocation,dnum)
VALUES (30, '온라인 마케팅 강화를 위한 전략 수립', '부산', 1);
INSERT INTO project(pnumber,pname,plocation,dnum)
VALUES (40, '보안 강화를 위한 인프라 구축', '대전', 5);
SELECT * FROM project;

-- 참여테이블 데이터 입력
INSERT INTO works_on (essn,pno,hours) VALUES ('444889999',10,20.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('444889999',40,10.5);
INSERT INTO works_on (essn,pno,hours) VALUES ('987987987',10,5.5);
INSERT INTO works_on (essn,pno,hours) VALUES ('987987987',20,20.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('987987987',40,5.5);
INSERT INTO works_on (essn,pno,hours) VALUES ('333445555',30,20.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('123456789',10,25.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('999887777',20,10.5);
INSERT INTO works_on (essn,pno,hours) VALUES ('999887777',10,5.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('987654321',40,5.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('666884444',40,20.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('453453453',10,20.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('888665555',30,25.0);
INSERT INTO works_on (essn,pno,hours) VALUES ('987654321',10,5.0);
SELECT * FROM works_on;

-- 부양가족 테이블 데이터 입력
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('333445555','김명희','F','1967/10/13','배우자');
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('333445555','강동진','M','1995/04/05','아들');
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('333445555','강수진','F','1998/05/05','딸');
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('987987987','남희정','F','1970/07/09','배우자');
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('987987987','김다래','F','2000/02/21','딸');
INSERT INTO dependent(essn,dpntname,sex,bdate,relation)
VALUES ('444889999','한명숙','F','1975/04/06','배우자');
SELECT * FROM dependent;
