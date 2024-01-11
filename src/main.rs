use abes_nice_things::{ai::*, prelude::*, OnceLockMethod};

use std::time::Duration;
use std::ops::{Not, RangeInclusive};
use std::thread::JoinHandle;

const MAX_CHANGE: RangeInclusive<f32> = -0.1..=0.1;

static END: OnceLockMethod<JoinHandle<()>> = OnceLockMethod::new(&|| -> JoinHandle<()> {
    return std::thread::spawn(|| {
        input();
    })
});

fn main() {
    // Getting run settings
    println!("Nodes:");
    let nodes: usize = input().parse().unwrap();
    println!("Layers:");
    let layers: usize = input().parse().unwrap();
    println!("Delay(ms):");
    let delay: u64 = input().parse().unwrap();
    let delay: Duration = Duration::from_millis(delay);
    END.init();
    Trainer::new()
    .nodes(nodes)
    .layers(layers)
    .nets(2)
    .delay(delay)
    .inputs(64)
    .outputs(4096)
    .max_change_inclusive(MAX_CHANGE)
    .cond_method(&|_| -> bool {
        END.get().as_ref().unwrap().is_finished()
    })
    .sequential()
    .train(|nets| {
        let mut game: Game = Game::new();
        let mut start: Pos;
        let mut end: Pos;
        for move_count in 0..60 {
            nets[0].inputs = game.get_values(Team::White);
            let outputs = nets[0].get_outputs();
            (start, end) = output_to_move(&game, Team::White, &outputs);
            assert!(game.play(&start, &end));
            if let Some(team) = game.end_check() {
                match team {
                    Team::White => {
                        nets[0].score = 15.0 - (move_count as f32)/10.0;
                        nets[1].score = (move_count as f32)/10.0;
                        break
                    }
                    Team::Black => {
                        panic!("Black won on white's turn");
                    }
                }
            }
            nets[1].inputs = game.get_values(Team::Black);
            (start, end) = output_to_move(&game, Team::White, &nets[1].get_outputs());
            assert!(game.play(&start, &end));
            if let Some(team) = game.end_check() {
                match team {
                    Team::White => {
                        panic!("White won on black's turn");
                    }
                    Team::Black => {
                        nets[0].score = (move_count as f32)/10.0;
                        nets[1].score = 15.0 - (move_count as f32)/10.0;
                        break
                    }
                }
            }
        }
        debug!({
            if nets[0].score != 0.0 {
                assert_eq!(nets[0].score + nets[1].score, 15.0, "Net scores did not add to 15\nWhite: {}\n Black {}", nets[0].score, nets[1].score);
            }
        });
        println!(
            "White: {}\nBlack: {}",
            nets[0].score,
            nets[1].score
        );
    });
}

fn get_character(value: &Option<Piece>) -> String {
    if let None = value {
        return " ".to_string();
    }
    match value.unwrap() {
        Piece::Pawn(piece) => match piece.team {
            Team::White => return "♙".to_string(),
            Team::Black => return "♟︎".to_string(),
        },
        Piece::Knight(piece) => match piece.team {
            Team::White => return "♘".to_string(),
            Team::Black => return "♞".to_string(),
        },
        Piece::Rook(piece) => match piece.team {
            Team::White => return "♖".to_string(),
            Team::Black => return "♜".to_string(),
        },
        Piece::Bishop(piece) => match piece.team {
            Team::White => return "♗".to_string(),
            Team::Black => return "♝".to_string(),
        },
        Piece::Queen(piece) => match piece.team {
            Team::White => return "♕".to_string(),
            Team::Black => return "♛".to_string(),
        },
        Piece::King(piece) => match piece.team {
            Team::White => return "♔".to_string(),
            Team::Black => return "♚".to_string(),
        },
    }
}

type Board = [[Option<Piece>; 8]; 8];
struct Game {
    board: Board,
    team: Team,
}
impl Game {
    /// If the place didn't exist, returns error.
    /// If there was something there, returns true.
    fn add(&mut self, pos: &Pos, new: &Piece) -> Result<bool, &str> {
        if let Some(row) = self.board.get_mut(pos.x) {
            if let Some(target) = row.get_mut(pos.y) {
                let mut out: bool = false;
                if let Some(_) = target {
                    out = true;
                }
                *target = Some(*new);
                return Ok(out);
            }
            return Err("Invalid column");
        }
        return Err("Invalid row");
    }
    /// If the place didn't exist, returns error.
    /// If there was something there, returns true.
    fn remove(&mut self, pos: &Pos) -> Result<bool, &str> {
        if let Some(row) = self.board.get_mut(pos.x) {
            if let Some(target) = row.get_mut(pos.y) {
                let mut out: bool = false;
                if let Some(_) = target {
                    out = true;
                }
                *target = None;
                return Ok(out);
            }
            return Err("Invalid column");
        }
        return Err("Invalid row");
    }
    fn new() -> Game {
        Game {
            board: [
                [
                    Some(Piece::Rook(Rook {
                        pos: Pos::new(0, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(0, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(0, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Rook(Rook {
                        pos: Pos::new(0, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Knight(Knight {
                        pos: Pos::new(1, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(1, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(1, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Knight(Knight {
                        pos: Pos::new(1, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Bishop(Bishop {
                        pos: Pos::new(2, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(2, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(2, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Bishop(Bishop {
                        pos: Pos::new(2, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::King(King {
                        pos: Pos::new(3, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(3, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(3, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Queen(Queen {
                        pos: Pos::new(3, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Queen(Queen {
                        pos: Pos::new(4, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(4, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(4, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::King(King {
                        pos: Pos::new(4, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Bishop(Bishop {
                        pos: Pos::new(5, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(5, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(5, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Bishop(Bishop {
                        pos: Pos::new(5, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Knight(Knight {
                        pos: Pos::new(6, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(6, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(6, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Knight(Knight {
                        pos: Pos::new(6, 7),
                        team: Team::Black,
                    })),
                ],
                [
                    Some(Piece::Rook(Rook {
                        pos: Pos::new(7, 0),
                        team: Team::White,
                    })),
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(7, 1),
                        team: Team::White,
                    })),
                    None,
                    None,
                    None,
                    None,
                    Some(Piece::Pawn(Pawn {
                        pos: Pos::new(7, 6),
                        team: Team::Black,
                    })),
                    Some(Piece::Rook(Rook {
                        pos: Pos::new(7, 7),
                        team: Team::Black,
                    })),
                ],
            ],
            team: Team::White,
        }
    }
    fn end_check(&self) -> Option<Team> {
        let mut white_king_found: bool = false;
        let mut black_king_found: bool = false;
        for row in self.board.iter() {
            for target in row.iter() {
                if let Some(piece) = target {
                    if let Piece::King(king) = piece {
                        match king.team {
                            Team::White => white_king_found = true,
                            Team::Black => black_king_found = true,
                        }
                    }
                }
            }
        }
        if white_king_found == black_king_found {
            if !white_king_found {
                panic!("Both kings are missing")
            }
        } else if !white_king_found {
            return Some(Team::Black);
        } else if !black_king_found {
            return Some(Team::White);
        }
        return None;
    }
    fn draw(&self) {
        let board = self.board_swap();
        self.draw_top_row();
        self.draw_piece_row(&board, 7);
        self.draw_empty_row();
        self.draw_piece_row(&board, 6);
        self.draw_empty_row();
        self.draw_piece_row(&board, 5);
        self.draw_empty_row();
        self.draw_piece_row(&board, 4);
        self.draw_empty_row();
        self.draw_piece_row(&board, 3);
        self.draw_empty_row();
        self.draw_piece_row(&board, 2);
        self.draw_empty_row();
        self.draw_piece_row(&board, 1);
        self.draw_empty_row();
        self.draw_piece_row(&board, 0);
        self.draw_bottom_row();
    }
    fn draw_top_row(&self) {
        println!("╔═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╗");
    }
    fn draw_bottom_row(&self) {
        println!("╚═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╝");
    }
    fn draw_empty_row(&self) {
        println!("╠═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╣");
    }
    fn draw_piece_row(&self, board: &Vec<Vec<Option<Piece>>>, index: usize) {
        let mut out: String = "║".to_string();
        for target in board.get(index).unwrap().iter() {
            let mut temp: String = " ".to_string();
            temp += get_character(target).as_str();
            temp += " ║";
            out += &temp;
        }
        println!("{out}");
    }
    fn board_swap(&self) -> Vec<Vec<Option<Piece>>> {
        let mut out: Vec<Vec<Option<Piece>>> = Vec::new();
        let iter: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        for x in iter.iter() {
            let mut temp: Vec<Option<Piece>> = Vec::new();
            for y in iter.iter() {
                temp.push(self.board[*y][*x])
            }
            out.push(temp)
        }
        return out;
    }
    fn valid(&self, start: &Pos, end: &Pos) -> bool {
        match self.get_pos(start).unwrap() {
            Some(piece) => return piece.valid(end, &self.board),
            None => return false,
        }
    }
    fn play(&mut self, start: &Pos, end: &Pos) -> bool {
        let mut piece: Piece = self.get_pos(start).unwrap().unwrap();
        piece.set_pos(end);
        assert!(self.remove(start).unwrap());
        self.add(end, &piece).unwrap();
        self.team = !self.team;
        return true;
    }
    /// Err if pos doesn't exist
    fn get_pos(&self, pos: &Pos) -> Result<Option<Piece>, &str> {
        match self.board.get(pos.x) {
            Some(row) => match row.get(pos.y) {
                Some(target) => return Ok(*target),
                None => return Err("Invalid column"),
            },
            None => return Err("Invalid row"),
        }
    }
    fn get_values(&self, team: Team) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for row in self.board.iter() {
            for target in row.iter() {
                if let Some(piece) = target {
                    out.push(piece.value(team));
                    continue;
                }
                out.push(0.0);
            }
        }
        return out;
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
struct Pos {
    x: usize,
    y: usize,
}
impl Pos {
    fn new(x: usize, y: usize) -> Pos {
        return Pos { x, y };
    }
    fn offset_i(&self, x: isize, y: isize) -> Option<Pos> {
        let new_x: isize = (self.x as isize) + x;
        let new_y: isize = (self.y as isize) + y;
        if new_x < 0 {
            return None;
        }
        if new_y < 0 {
            return None;
        }
        if new_x > 7 {
            return None;
        }
        if new_y > 7 {
            return None;
        }
        return Some(Pos::new(new_x as usize, new_y as usize));
    }
}
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
enum Team {
    #[default]
    White,
    Black,
}
impl Not for Team {
    type Output = Team;
    fn not(self) -> Self::Output {
        match self {
            Team::White => Team::Black,
            Team::Black => Team::White,
        }
    }
}
impl Team {
    fn direction(&self) -> isize {
        match self {
            Team::White => return 1,
            Team::Black => return -1,
        }
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Piece {
    Pawn(Pawn),
    Knight(Knight),
    Rook(Rook),
    Bishop(Bishop),
    Queen(Queen),
    King(King),
}
impl Piece {
    fn team(&self) -> Team {
        match self {
            Piece::Pawn(target) => return target.team,
            Piece::Knight(target) => return target.team,
            Piece::Rook(target) => return target.team,
            Piece::Bishop(target) => return target.team,
            Piece::Queen(target) => return target.team,
            Piece::King(target) => return target.team,
        }
    }
    fn pos(&self) -> Pos {
        match self {
            Piece::Pawn(target) => return target.pos,
            Piece::Knight(target) => return target.pos,
            Piece::Rook(target) => return target.pos,
            Piece::Bishop(target) => return target.pos,
            Piece::Queen(target) => return target.pos,
            Piece::King(target) => return target.pos,
        }
    }
    fn set_pos(&mut self, new: &Pos) {
        match self {
            Piece::Pawn(piece) => {
                piece.pos = *new;
            }
            Piece::Knight(piece) => {
                piece.pos = *new;
            }
            Piece::Rook(piece) => {
                piece.pos = *new;
            }
            Piece::Bishop(piece) => {
                piece.pos = *new;
            }
            Piece::Queen(piece) => {
                piece.pos = *new;
            }
            Piece::King(piece) => {
                piece.pos = *new;
            }
        }
    }
    fn value(&self, team: Team) -> f32 {
        let value: f32;
        match self {
            Piece::Pawn(_) => value = Pawn::VALUE,
            Piece::Knight(_) => value = Knight::VALUE,
            Piece::Rook(_) => value = Rook::VALUE,
            Piece::Bishop(_) => value = Bishop::VALUE,
            Piece::Queen(_) => value = Queen::VALUE,
            Piece::King(_) => value = King::VALUE,
        }
        if !(self.team() == team) {
            return -value;
        }
        return value;
    }
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        match self {
            Piece::Pawn(piece) => return piece.valid(end, board),
            Piece::Knight(piece) => return piece.valid(end, board),
            Piece::Rook(piece) => return piece.valid(end, board),
            Piece::Bishop(piece) => return piece.valid(end, board),
            Piece::Queen(piece) => return piece.valid(end, board),
            Piece::King(piece) => return piece.valid(end, board),
        }
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Pawn {
    pos: Pos,
    team: Team,
}
impl Pawn {
    const VALUE: f32 = 1.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        let direction: isize = self.team.direction();
        if self.valid_forward(end, board, direction) {
            return true;
        }
        if self.valid_diagonal(end, board, direction, -1) {
            return true;
        }
        if self.valid_diagonal(end, board, direction, 1) {
            return true;
        }
        return false;
    }
    fn valid_forward(&self, end: &Pos, board: &Board, direction: isize) -> bool {
        let target: Pos = self.directional_offset(0, 1, direction);
        if !(target == *end) {
            return false;
        }
        let wrapped: Option<Piece>;
        match get_pos(board, end) {
            Ok(value) => wrapped = *value,
            Err(_) => return false,
        }
        match wrapped {
            Some(_) => return false,
            None => return true,
        }
    }
    fn valid_diagonal(&self, end: &Pos, board: &Board, direction: isize, x_offset: isize) -> bool {
        let target: Pos = self.directional_offset(x_offset, 1, direction);
        if !(target == *end) {
            return false;
        }
        let wrapped: Option<Piece>;
        match get_pos(board, end) {
            Ok(value) => wrapped = *value,
            Err(_) => return false,
        }
        match wrapped {
            Some(piece) => {
                if self.team == piece.team() {
                    return false;
                }
                return true;
            }
            None => return false,
        }
    }
    fn directional_offset(&self, x: isize, y: isize, direction: isize) -> Pos {
        let mut out: Pos = self.pos.clone();
        out.x = (out.x as isize + x) as usize;
        let y: isize = y * direction;
        out.y = (out.y as isize + y) as usize;
        return out;
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Knight {
    pos: Pos,
    team: Team,
}
impl Knight {
    const VALUE: f32 = 2.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        let x_offset: isize = ((self.pos.x as isize) - (end.x as isize)).abs();
        let y_offset: isize = ((self.pos.y as isize) - (end.y as isize)).abs();
        let piece: Option<Piece>;
        match get_pos(board, end) {
            Ok(value) => piece = *value,
            Err(_) => return false,
        }
        if let Some(piece) = piece {
            if self.team == piece.team() {
                return false;
            }
        }
        if x_offset == 1 {
            if y_offset == 2 {
                return true;
            }
        } else if x_offset == 2 {
            if y_offset == 1 {
                return true;
            }
        }
        return false;
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Rook {
    pos: Pos,
    team: Team,
}
impl Rook {
    const VALUE: f32 = 3.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        return rbq_hor(&self.pos, end, board);
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Bishop {
    pos: Pos,
    team: Team,
}
impl Bishop {
    const VALUE: f32 = 4.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        return rbq_dia(&self.pos, end, board);
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Queen {
    pos: Pos,
    team: Team,
}
impl Queen {
    const VALUE: f32 = 5.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        if rbq_dia(&self.pos, end, board) {
            return true;
        }
        if rbq_hor(&self.pos, end, board) {
            return true;
        } else {
            return false;
        }
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct King {
    pos: Pos,
    team: Team,
}
impl King {
    const VALUE: f32 = 6.0;
    fn valid(&self, end: &Pos, board: &Board) -> bool {
        let x_offset: isize = (self.pos.x as isize) - (end.x as isize);
        let y_offset: isize = (self.pos.y as isize) - (end.y as isize);
        if x_offset.abs() > 1 {
            return false;
        }
        if y_offset.abs() > 1 {
            return false;
        }
        if &self.pos == end {
            return false;
        }
        let piece: Option<Piece>;
        match get_pos(board, end) {
            Ok(value) => piece = *value,
            Err(_) => return false,
        }
        if let Some(piece) = piece {
            if piece.team() == self.team {
                return false;
            }
        }
        return true;
    }
}
fn rbq_dia(start: &Pos, end: &Pos, board: &Board) -> bool {
    let x_offset: isize = (start.x as isize) - (end.x as isize);
    let y_offset: isize = (start.y as isize) - (end.y as isize);
    if !x_offset.abs() == y_offset.abs() {
        return false;
    }
    if start == end {
        return false;
    }
    let x_inc: isize;
    let y_inc: isize;
    if start.x > end.x {
        x_inc = -1
    } else if start.x < end.x {
        x_inc = 1
    } else {
        return false;
    }
    if start.y > end.y {
        y_inc = -1
    } else if start.y < end.y {
        y_inc = 1
    } else {
        return false;
    }
    return rbq_loop(start, end, board, x_inc, y_inc);
}
fn rbq_hor(start: &Pos, end: &Pos, board: &Board) -> bool {
    if !start.x == end.x {
        if !start.y == end.y {
            return false;
        }
    }
    if start == end {
        return false;
    }
    let mut x_inc: isize = 0;
    let mut y_inc: isize = 0;
    if start.x > end.x {
        x_inc = -1
    } else if start.x < end.x {
        x_inc = 1
    } else if start.y > end.y {
        y_inc = -1
    } else if start.y < end.y {
        y_inc = 1
    } else {
        return false;
    }
    return rbq_loop(start, end, board, x_inc, y_inc);
}
fn rbq_loop(start: &Pos, end: &Pos, board: &Board, x_inc: isize, y_inc: isize) -> bool {
    let mut pos: Pos = start.clone();
    let start_piece: Piece;
    match get_pos(board, start).unwrap() {
        Some(piece) => start_piece = *piece,
        None => return false,
    }
    loop {
        match pos.offset_i(x_inc, y_inc) {
            Some(value) => pos = value,
            None => return false,
        }
        if let Some(piece) = get_pos(board, &pos).unwrap() {
            if piece.team() == start_piece.team() {
                return false;
            }
            if end == &pos {
                return true;
            }
            return false;
        }
        if end == &pos {
            return true;
        }
    }
}
/// Err if pos doesn't exist
fn get_pos<'a>(board: &'a Board, pos: &Pos) -> Result<&'a Option<Piece>, &'a str> {
    match board.get(pos.x) {
        Some(row) => match row.get(pos.y) {
            Some(target) => return Ok(target),
            None => return Err("Invalid column"),
        },
        None => return Err("Invalid row"),
    }
}

fn output_to_move(game: &Game, team: Team, outputs: &[f32]) -> (Pos, Pos) {
    let mut outputs: Vec<(f32, usize)> = outputs.iter().enumerate().map(|(index, value)| -> (f32, usize) {
        return (*value, index)
    }).collect();
    outputs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    for (_, index) in outputs.iter() {
        let indexs = index_to_indexs(index);
        let start = index_to_pos(&indexs.0);
        let end = index_to_pos(&indexs.1);
        match game.get_pos(&start).unwrap() {
            Some(piece) => {
                if !(piece.team() == team) {
                    continue
                }
            }
            None => {
                continue
            }
        }
        if let Some(piece) = game.get_pos(&end).unwrap() {
            if piece.team() == team {
                continue
            }
        }
        if game.valid(&start, &end) {
            return (start, end)
        }
    }
    panic!("No move")
}
fn index_to_indexs(index: &usize) -> (usize, usize) {
    return (index/64, index%64);
}
fn index_to_pos(index: &usize) -> Pos {
    return Pos::new(index / 8, index % 8);
}