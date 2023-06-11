CREATE DATABASE company;
USE company;

-- 사원테이블 정의
CREATE TABLE employee
(
   ename	VARCHAR(15) NOT NULL,
   ssn		CHAR(9) PRIMARY KEY,
   address	VARCHAR(30),
   salary  	INT,
   sex		CHAR(1) NOT NULL CHECK (sex IN ('M','F')),
   bdate  	DATETIME,	
   dno		SMALLINT,	
   superssn	CHAR(9)
);

-- 부서테이블 정의
CREATE TABLE department 
(
  dnumber  SMALLINT PRIMARY KEY,	
  dname    VARCHAR(15) NOT NULL UNIQUE,
  mgrssn   CHAR(9),
  mgrstartdate DATETIME
);

-- 부서위치테이블 정의
CREATE TABLE dept_locations
(
  dnumber    SMALLINT,
  dlocations VARCHAR(15),
  CONSTRAINT PRIMARY KEY (dnumber, dlocations)
);

-- 프로젝트테이블 정의
CREATE TABLE project
(
  pnumber   INT PRIMARY KEY,	
  pname     VARCHAR(50) NOT NULL UNIQUE,
  plocation VARCHAR(15),
  dnum      SMALLINT NOT NULL
);

-- (프로젝트)참여테이블 정의
CREATE TABLE works_on
(
   essn  CHAR(9),
   pno   INT,	
   hours DECIMAL(3,1),
   CONSTRAINT PRIMARY KEY (essn, pno)
);

-- 부양가족테이블 정의
CREATE TABLE dependent
(
  essn     CHAR(9),
  dpntname VARCHAR(15) NOT NULL,
  sex      CHAR(1) CHECK (sex IN ('M','F')),
  bdate    DATETIME,
  relation VARCHAR(10) NOT NULL,
  CONSTRAINT PRIMARY KEY (essn, dpntname)
);

-- 외래키정의
ALTER TABLE employee
ADD CONSTRAINT fk_emp_dept FOREIGN KEY (dno) REFERENCES department(dnumber);

ALTER TABLE employee
ADD CONSTRAINT fk_emp_emp FOREIGN KEY (superssn) REFERENCES employee(ssn);

ALTER TABLE department
ADD CONSTRAINT fk_dept_emp FOREIGN KEY (mgrssn) REFERENCES employee(ssn);

ALTER TABLE dept_locations
ADD CONSTRAINT fk_dl_dept FOREIGN KEY (dnumber) REFERENCES department(dnumber);

ALTER TABLE project
ADD CONSTRAINT fk_pj_dept FOREIGN KEY (dnum) REFERENCES department(dnumber);

ALTER TABLE works_on
ADD CONSTRAINT fk_wo_emp FOREIGN KEY (essn) REFERENCES employee(ssn);

ALTER TABLE works_on
ADD CONSTRAINT fk_wo_pj FOREIGN KEY (pno) REFERENCES project(pnumber);

ALTER TABLE DEPENDENT
ADD CONSTRAINT fk_dpnt_emp FOREIGN KEY (essn) REFERENCES employee(ssn);
