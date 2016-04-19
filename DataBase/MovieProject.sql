CREATE DATABASE MOVIES_SQL;
USE MOVIES_SQL;
/*movie(title, year, director, studio, category)

star(title, year, stage)

director(ref_name, last_n, first_n)

studio(name, city, founder, country, first)

actor(stage_name, gender, dob, type)*/
CREATE TABLE director (
	ref_name       varchar(30) 	       NOT NULL,
	last_n         varchar(20)         NOT NULL,
    first_n        varchar(20)         NULL,
    PRIMARY KEY (ref_name,last_n)
	);
CREATE TABLE studio (
	name           varchar(30) 	       NOT NULL,
	city           varchar(30)         NULL,
    founder        varchar(30)         NULL,
    country        varchar(30)         NOT NULL,
    first_n        INT(4)              NULL,
    PRIMARY KEY (name, country)
	);
CREATE TABLE actor (
    stage          varChar(30)     NOT NULL,
    gen            CHAR(1)         NULL,
    dob            INT(4)          NULL,
    type           varChar(30)     NULL,
    PRIMARY KEY (stage)
	);

CREATE TABLE movie (
    title        Char(40) 	            NOT NULL,
	year         INT(4) 	            NOT NULL,
	director     VARChar(50)            NOT NULL,
    studio       VARChar(30)            NULL,
    category     VARChar(30)            NULL,
    PRIMARY KEY (title, year),
	FOREIGN KEY (studio) REFERENCES studio (name)
       ON UPDATE CASCADE
	);
CREATE TABLE star (
    title        Char(40) 	            NOT NULL,
	year         INT(4) 	            NOT NULL,
	stage        VARChar(30)            NOT NULL,
    PRIMARY KEY (title, year, stage),
    FOREIGN KEY (stage) REFERENCES actor (stage)
    ON UPDATE CASCADE,
    FOREIGN KEY (title, year) REFERENCES movie (title,year)
    ON UPDATE CASCADE
	);
    
/*director(ref_name, last_n, first_n)*/
INSERT INTO director VALUES (
	'Aaron', 'Aaron', 'Paul');
INSERT INTO director VALUES (
	'Abel',	'Abel',	'Jeanne');
INSERT INTO director VALUES (
	'Abbott', 'Abbott',	'George' );
INSERT INTO director VALUES (
	'Abrahams', 'Abrahams',	NULL );
INSERT INTO director VALUES (
	'J.Abrahams', 'Abrahams', 'Jim' );
INSERT INTO director VALUES (
	'Jeffrey.Abrams', 'Abrams',	'Jeffrey' );
INSERT INTO director VALUES (
	'Abramson',	'Abramson',	'Neil' );
INSERT INTO director VALUES (
	'Abduladze', 'Abduladze', 'Tengiz' );
INSERT INTO director VALUES (
	'Acub',	'Acub',	'Nya. Abbas' );
INSERT INTO director VALUES (
	'Ackerman',	'Ackerman',	'Robert Allan');

/*studio(name, city, founder, country, first)*/
INSERT INTO studio VALUES (
	'Ph.enakistiscope',	'Paris', 'France', 1832, 'Plateau');
INSERT INTO studio VALUES (
	'Zootrope',	NULL, NULL,	1834, 'W.G.Horne' );
INSERT INTO studio VALUES (
	'Phenakistiscope', 'Paris',	'France', 1835,	'CharlesChevalier' );
INSERT INTO studio VALUES (
	'Lanorthoscope', 'Paris', 'France',	1836, 'Plateau');
INSERT INTO studio VALUES (
	'Photobioscope', NULL, NULL, 1867, 'Bonnelliot,HenryCook');
INSERT INTO studio VALUES (
	'Pocket Kinematograph',	NULL, NULL,	1868, 'J.B.Linnett' );
INSERT INTO studio VALUES (
	'Zootrope',	'Paris', 'France', 1876, 'EmileReynaud');
INSERT INTO studio VALUES (
	'Thaumatrope', 'Paris',	'France', 1877,	'DocteurPan');
INSERT INTO studio VALUES (
	'Life Projection Wheel', 'Stanford',	'USA', 1880, 'E.Muybridge');
INSERT INTO studio VALUES (
	'Schnellseher',	NULL, 'Germany', 1887,	'O.Ansch."utz');

/*actor(stage_name, gender, dob, type)*/
INSERT INTO actor VALUES (
	'Bud Abbott', 'M',	1895, 'straight' );
INSERT INTO actor VALUES (
	'George Abbott', 'M', 1887,	'playwright, producer' );
INSERT INTO actor VALUES (
	'John Abbott', 'M',	1905, 'staring eyes, eccentric parts' );
INSERT INTO actor VALUES (
	'Philip Abbott', 'M', 1923,	'second lead');
INSERT INTO actor VALUES (
	'Walter Abel', 'M',	1898, 'harrassed hero');
INSERT INTO actor VALUES (
	'Joss Ackland',	'M', 1928, 'larger-than-life persolity' );
INSERT INTO actor VALUES (
	'Rodolfo Acosta', 'M', 1920, 'cold-eyed');
INSERT INTO actor VALUES (
	'Eddie Acuff', 'M',	1902, 'supporting comedian');
INSERT INTO actor VALUES (
	'Jean Adair', 'F',	1872, 'sweet aunt' );
INSERT INTO actor VALUES (
	'Alfred Adam', 'M',	1909, 'weak villainous');
    

INSERT INTO movie VALUES (
	'Always Tell Your Wife', 1922, 'Se.Hicks', 'Ph.enakistiscope', 'Dram' );
INSERT INTO movie VALUES (
	'Woman to Woman', 1922, 'Hitchcock', 'Zootrope', 'Dram' );
INSERT INTO movie VALUES (
	'The Pleasure Garden', 1925, 'Hitchcock', 'Phenakistiscope', 'Dram' );
INSERT INTO movie VALUES (
	'The Mountain Eagle', 1926,	'Hitchcock', 'Lanorthoscope', 'Dram' );
INSERT INTO movie VALUES (
	'The Lodger: A Story of The London Fog', 1926, 'Hitchcock', 'Photobioscope', 'Susp' );
INSERT INTO movie VALUES (
	'Downhill',	1927, 'Hitchcock', 'Pocket Kinematograph', 'Susp' );
INSERT INTO movie VALUES (
	'Easy Virtue', 1927, 'Hitchcock', 'Zootrope', 'Susp' );
INSERT INTO movie VALUES (
	'The Ring',	1927, 'Hitchcock', 'Thaumatrope', 'Susp' );
INSERT INTO movie VALUES (
	'The Farmers Wife', 1928, 'Hitchcock', 'Life Projection Wheel', 'Susp' );
INSERT INTO movie VALUES (
	'Champagne', 1928, 'Hitchcock', 'Schnellseher', 'Romt' );


INSERT INTO star VALUES (
	'Always Tell Your Wife', 1922, 'Bud Abbott' );
INSERT INTO star VALUES (
	'Woman to Woman', 1922, 'John Abbott' );
INSERT INTO star VALUES (
	'The Pleasure Garden', 1925, 'Joss Ackland' );
INSERT INTO star VALUES (
	'The Mountain Eagle', 1926, 'Rodolfo Acosta' );
INSERT INTO star VALUES (
	'The Lodger: A Story of The London Fog', 1926, 'Eddie Acuff' );
INSERT INTO star VALUES (
	'Downhill',	1927, 'Jean Adair');
INSERT INTO star VALUES (
	'Easy Virtue', 1927, 'Alfred Adam' );
INSERT INTO star VALUES (
	'The Ring',	1927, 'Eddie Acuff' );
INSERT INTO star VALUES (
	'The Farmers Wife', 1928, 'Jean Adair' );
INSERT INTO star VALUES (
	'Champagne', 1928, 'Alfred Adam' );
    
select * from star;
select * from director;
select * from movie;
select * from studio;
select * from actor;