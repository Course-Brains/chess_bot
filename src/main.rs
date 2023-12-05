use std::thread;
use std::{
    io::{stdin, stdout, Write},
    ops::RangeInclusive,
    sync::OnceLock,
};
use std::time::Duration;
use rand::{thread_rng, Rng};

const DEFAULT_WEIGHT: f32 = 1.0;
const WEIGHT_RANGE: RangeInclusive<f32> = -10.0..=10.0;
const MAX_CHANGE: RangeInclusive<f32> = -1.0..=1.0;

// End of run statistics
static mut HIGHEST_BLACK_SCORE: f32 = 0.0;
static mut HIGHEST_WHITE_SCORE: f32 = 0.0;

// Run settings
static NODES: OnceLock<usize> = OnceLock::new();
static LAYERS: OnceLock<usize> = OnceLock::new();
static NETS: OnceLock<usize> = OnceLock::new();
fn main() {
    // Getting run settings
    println!("Nodes:");
    let nodes: usize = input().parse().unwrap();
    NODES.set(nodes).unwrap();
    println!("Layers:");
    let layers: usize = input().parse().unwrap();
    LAYERS.set(layers).unwrap();
    println!("Nets:");
    let nets: usize = input().parse().unwrap();
    NETS.set(nets).unwrap();
    println!("Delay(ms):");
    let delay: u64 = input().parse().unwrap();
    let delay: Duration = Duration::from_millis(delay);

    println!("Print modulo");
    let print_modulo: usize = input().parse().unwrap();

    let handle = thread::spawn(|| {
        input();
    });
    let mut trainer: Trainer = Trainer::new(*NODES.get().unwrap(), *LAYERS.get().unwrap(), *NETS.get().unwrap());
    trainer.randomize_weights();
    let mut iteration: usize = 0;
    loop {
        thread::sleep(delay);
        if iteration%print_modulo == 0 {
            println!("Iteration: {}", iteration);
        }
        trainer.generate_score();
        trainer.find_best();
        trainer.update_weights();
        trainer.randomize_weights();
        iteration += 1;
        if handle.is_finished() {
            break
        }
    }
    println!("Highest white score: {}", unsafe { HIGHEST_WHITE_SCORE });
    println!("Highest black score: {}", unsafe { HIGHEST_BLACK_SCORE });
}

fn input() -> String {
    let mut string: String = String::new();
    let _ = stdout().flush();
    stdin().read_line(&mut string).unwrap();
    if let Some('\n') = string.chars().next_back() {
        string.pop();
    } else if let Some('\r') = string.chars().next_back() {
        string.pop();
    }
    return string;
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

#[derive(Default, Debug)]
struct Trainer {
    nets: Vec<(Net, f32)>,
    best: Net,
    best_score: f32,
}
impl Trainer {
    fn generate_score(&mut self) {
        let mut first: (Net, f32) = (Default::default(), 0.0);
        let mut previous: &mut Net = &mut first.0;
        let mut previous_score: &mut f32 = &mut first.1;
        for (index, (net, score)) in self.nets.iter_mut().enumerate() {
            if index%2 == 0 {
                previous = net;
                previous_score = score;
                continue
            }
            let mut game: Game = Game::new();
            let mut time_out: bool = false;
            let mut bad_play: bool = false;
            let mut num_iter: usize = 0;
            for iter in 0..40 {
                if !net.do_play(&mut game, Team::White) {
                    bad_play = true;
                    num_iter = iter;
                    break
                }
                if let Some(out) = game.end_check() {
                    match out {
                        Team::White => {
                            *score = 10.0 - (iter as f32/10.0);
                            time_out = true;
                            println!("White win");
                            break
                        }
                        Team::Black => {
                            *score = iter as f32/10.0;
                            time_out = true;
                            println!("Black win");
                            break
                        }
                    }
                }
                if !previous.do_play(&mut game, Team::Black) {
                    bad_play = true;
                    num_iter = iter;
                    break
                }
                if let Some(out) = game.end_check() {
                    match out {
                        Team::White => {
                            *score = iter as f32/10.0;
                            time_out = true;
                            println!("White win");
                            break
                        }
                        Team::Black => {
                            *score = 10.0 - (iter as f32/10.0);
                            time_out = true;
                            println!("Black win");
                            break
                        }
                    }
                }
            }
            if bad_play {
                *score = num_iter as f32/ 10.0;
            }
            if time_out {
                *score = 0.0;
                *previous_score = 0.0;
            }
            // if *score > 0.1 || *previous_score > 0.1 {
            //     println!("Better score than 0.1:\nwhite: {}\nblack: {}", score, previous_score);
            //     panic!("YAY");
            // }
            if *score > unsafe { HIGHEST_WHITE_SCORE } {
                unsafe { HIGHEST_WHITE_SCORE = *score };
            }
            if *previous_score > unsafe { HIGHEST_BLACK_SCORE } {
                unsafe { HIGHEST_BLACK_SCORE = *previous_score };
            }
        }
    }
    fn find_best(&mut self) {
        for (net, score) in self.nets.iter() {
            if score > &self.best_score {
                self.best = net.clone();
                self.best_score = *score;
            }
        }
    }
    fn update_weights(&mut self) {
        for (net, _) in self.nets.iter_mut() {
            *net = self.best.clone();
        }
    }
    fn new(nodes: usize, layers: usize, nets: usize) -> Trainer {
        let mut out: Trainer = Trainer {
            nets: Vec::new(),
            best: Net::new(nodes, layers, 128, 64),
            best_score: 0.0,
        };
        for _ in 0..nets {
            out.nets.push((Net::new(nodes, layers, 128, 64), 0.0));
        }
        return out
    }
    fn randomize_weights(&mut self) {
        for (net, _) in self.nets.iter_mut() {
            net.randomize_weights()
        }
    }
}
#[derive(Default, Debug, Clone)]
struct Net {
    nodes: Vec<Node>,
    inputs: Vec<f32>,
}
impl Net {
    fn get_outputs(&self) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for node in self.nodes.iter() {
            out.push(node.get_value(&self.inputs))
        }
        return out;
    }
    fn new(nodes: usize, layers: usize, outputs: usize, inputs: usize) -> Net {
        let mut net: Net = Default::default();
        for _ in 0..outputs {
            net.nodes.push(Node::new(nodes, layers, inputs))
        }
        return net;
    }
    fn node_vec(&self) -> Vec<Node> {
        let mut list: Vec<Node> = Vec::new();
        for node in self.nodes.iter() {
            list.extend_from_slice(node.node_vec().as_slice())
        }
        return list;
    }
    fn node_iter_mut(&mut self) -> Vec<&mut Node> {
        let mut list: Vec<&mut Node> = Vec::new();
        for node in self.nodes.iter_mut() {
            list.extend(node.node_vec_mut())
        }
        return list;
    }
    fn randomize_weights(&mut self) {
        for node in self.nodes.iter_mut() {
            node.randomize_weights()
        }
    }
    fn do_play(&mut self, game: &mut Game, team: Team) -> bool {
        self.inputs = game.get_values(team);
        let outputs: Vec<f32> = self.get_outputs();
        let start: Pos = max_pos_from_list(&outputs[0..63]);
        let end: Pos = max_pos_from_list(&outputs[64..127]);
        return game.play(&start, &end)
    }
}
#[derive(Default, Debug, Clone)]
struct Node {
    inputs: Vec<(NodeType, f32)>,
}
impl Node {
    fn empty() -> Node {
        return Node { inputs: Vec::new() };
    }
    fn get_value(&self, inputs: &Vec<f32>) -> f32 {
        let mut value: f32 = 0.0;
        for (node, weight) in self.inputs.iter() {
            value += node.get_value(inputs) * weight
        }
        return value;
    }
    fn new(nodes: usize, layers: usize, inputs: usize) -> Node {
        if layers == 0 {
            let mut node: Node = Node::empty();
            for i in 0..inputs {
                node.inputs.push((NodeType::Input(i), DEFAULT_WEIGHT))
            }
            return node;
        }
        let mut node: Node = Node::empty();
        for i in 0..nodes {
            node.inputs.push((
                NodeType::Node(Node::new(nodes, layers - 1, inputs)),
                DEFAULT_WEIGHT,
            ))
        }
        return node;
    }
    fn node_vec(&self) -> Vec<Node> {
        let mut list: Vec<Node> = Vec::new();
        if let NodeType::Input(_) = self.inputs[0].0 {
            return list;
        }
        for (node, _) in self.inputs.iter() {
            list.extend_from_slice(node.unwrap_node().node_vec().as_slice())
        }
        return list;
    }
    fn node_vec_mut(&mut self) -> Vec<&mut Node> {
        let mut list: Vec<&mut Node> = Vec::new();
        if let NodeType::Input(_) = self.inputs[0].0 {
            return list;
        }
        for (node, _) in self.inputs.iter_mut() {
            list.extend(node.unwrap_node_mut().node_vec_mut())
        }
        return list;
    }
    fn randomize_weights(&mut self) {
        for (node, weight) in self.inputs.iter_mut() {
            let mut rng = thread_rng();
            *weight += rng.gen_range(MAX_CHANGE);
            if let NodeType::Node(node) = node {
                node.randomize_weights();
            }
        }
    }
}
#[derive(Debug, Clone)]
enum NodeType {
    Node(Node),
    Input(usize),
}
impl NodeType {
    fn get_value(&self, inputs: &Vec<f32>) -> f32 {
        match self {
            NodeType::Node(node) => return node.get_value(inputs),
            NodeType::Input(index) => return inputs[*index],
        }
    }
    fn unwrap_node(&self) -> &Node {
        if let NodeType::Node(node) = self {
            return node;
        }
        panic!("unwrap_node was called on a non node value")
    }
    fn unwrap_node_mut(&mut self) -> &mut Node {
        if let NodeType::Node(node) = self {
            return node;
        }
        panic!("unwrap_node_mut was called on a non node value")
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
        match get_pos(&self.board, start).unwrap() {
            Some(piece) => return piece.valid(end, &self.board),
            None => return false,
        }
    }
    fn play(&mut self, start: &Pos, end: &Pos) -> bool {
        if !self.valid(start, end) {
            return false;
        }
        let mut piece: Piece = get_pos(&self.board, start).unwrap().unwrap();
        piece.set_pos(end);
        assert!(self.remove(start).unwrap());
        self.add(end, &piece).unwrap();
        self.team = self.team.swap();
        return true;
    }
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
                    continue
                }
                out.push(0.0);
            }
        }
        return out
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
impl Team {
    const VALUE: f32 = 1.0;
    fn to_value(&self, team: &Team) -> f32 {
        if self == team {
            return Team::VALUE;
        } else {
            return -Team::VALUE;
        }
    }
    fn direction(&self) -> isize {
        match self {
            Team::White => return 1,
            Team::Black => return -1,
        }
    }
    fn swap(&self) -> Team {
        match self {
            Team::White => return Team::Black,
            Team::Black => return Team::White,
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
            Ok(value) => wrapped = value,
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
            Ok(value) => wrapped = value,
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
            Ok(value) => piece = value,
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
            Ok(value) => piece = value,
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
        Some(piece) => start_piece = piece,
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
fn get_pos(board: &Board, pos: &Pos) -> Result<Option<Piece>, String> {
    match board.get(pos.x) {
        Some(row) => match row.get(pos.y) {
            Some(target) => return Ok(*target),
            None => return Err("Invalid column".to_string()),
        },
        None => return Err("Invalid row".to_string()),
    }
}
fn max_pos_from_list(list: &[f32]) -> Pos {
    let mut out: Pos = Pos::new(0,0);
    let mut highest: f32 = 0.0;
    for (index, value) in list.iter().enumerate() {
        if value > &highest {
            out = Pos::new((index+1)/8, (index+1)%8);
            highest = *value;
        }
    }
    return out
}

#[cfg(test)]
fn play(start: &Pos, end: &Pos, piece: Option<Piece>, target: Option<Piece>) -> Result<(), ()> {
    let mut game: Game = Game::new();
    if let None = game.get_pos(start).unwrap() {
        match piece {
            Some(piece) => game.board[start.x][start.y] = Some(piece),
            None => {
                panic!("No piece at start pos")
            }
        }
    }
    game.board[end.x][end.y] = target;
    if !game.play(start, end) {
        return Err(());
    }
    if let Some(_) = game.get_pos(start).unwrap() {
        panic!("Piece in previous pos")
    }
    if let None = game.get_pos(end).unwrap() {
        panic!("No piece in end pos")
    }
    return Ok(());
}
#[cfg(test)]
mod tests {
    #[allow(non_snake_case)]
    mod Game {
        use super::super::*;
        #[test]
        fn mpfl() {
            let mut start: [[f32; 8]; 8] = [[0.0; 8]; 8];
            start[4][4] = 10.0;
            let mut list: Vec<f32> = Vec::new();
            for row in start.iter() {
                for target in row.iter() {
                    list.push(*target)
                }
            }
            let out: Pos = max_pos_from_list(list.as_slice());
            assert_eq!(Pos::new(4,4), out, "Failed")
        }
        #[test]
        fn new_does_not_panic() {
            Game::new();
        }
        mod end_check {
            use super::super::super::*;
            #[test]
            fn returns_none_with_both_kings() {
                let game: Game = Game::new();
                assert_eq!(game.end_check(), None);
            }
            #[test]
            fn returns_White_on_only_white_king() {
                let mut game: Game = Game::new();
                assert!(
                    game.remove(&Pos { x: 4, y: 7 }).unwrap(),
                    "Black king space was empty"
                );
                assert_eq!(
                    game.end_check(),
                    Some(Team::White),
                    "end_check produced the wrong result"
                );
            }
            #[test]
            fn returns_Black_on_only_black_king() {
                let mut game: Game = Game::new();
                assert!(
                    game.remove(&Pos { x: 3, y: 0 }).unwrap(),
                    "White king space was empty"
                );
                assert_eq!(
                    game.end_check(),
                    Some(Team::Black),
                    "end_check produced the wrong result"
                );
            }
            #[test]
            #[should_panic(expected = "Both kings are missing")]
            fn panics_on_no_kings() {
                let mut game: Game = Game::new();
                assert!(
                    game.remove(&Pos { x: 4, y: 7 }).unwrap(),
                    "Black king space was empty"
                );
                assert!(
                    game.remove(&Pos { x: 3, y: 0 }).unwrap(),
                    "White king space was empty"
                );
                game.end_check();
            }
        }
        mod remove {
            use super::super::super::*;
            #[test]
            fn does_remove() {
                let mut game: Game = Game::new();
                assert!(
                    game.remove(&Pos { x: 0, y: 0 }).unwrap(),
                    "Piece wasn't there"
                );
                assert_eq!(game.board[0][0], None, "Piece wasn't removed");
            }
            #[test]
            fn returns_true_on_piece_removed() {
                let mut game: Game = Game::new();
                assert!(
                    match game.board[0][0] {
                        Some(_) => {
                            true
                        }
                        None => {
                            false
                        }
                    },
                    "No piece at 0,0"
                );
                assert!(game.remove(&Pos { x: 0, y: 0 }).unwrap(), "No piece at 0,0");
            }
            #[test]
            fn returns_false_on_no_piece() {
                let mut game: Game = Game::new();
                assert!(
                    match game.board[4][4] {
                        Some(_) => {
                            false
                        }
                        None => {
                            true
                        }
                    },
                    "Piece at 4,4"
                );
                assert!(!game.remove(&Pos { x: 4, y: 4 }).unwrap(), "Piece at 4,4");
            }
            mod returns_error_on_invalid_location {
                // Don't need to test for values under because it takes a usize
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "Invalid row")]
                fn over_row() {
                    let mut game: Game = Game::new();
                    game.remove(&Pos { x: 10, y: 0 }).unwrap();
                }
                #[test]
                #[should_panic(expected = "Invalid column")]
                fn over_column() {
                    let mut game: Game = Game::new();
                    game.remove(&Pos { x: 0, y: 10 }).unwrap();
                }
                #[test]
                #[should_panic(expected = "Invalid row")]
                fn over_both() {
                    let mut game: Game = Game::new();
                    game.remove(&Pos { x: 10, y: 10 }).unwrap();
                }
            }
        }
    }
    mod pieces {
        #[allow(non_snake_case)]
        mod Pawn {
            mod forward_one {
                use super::super::super::super::*;
                #[test]
                fn white_intended() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 1),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,1")
                    }
                    assert!(game.play(&Pos::new(1, 1), &Pos::new(1, 2)));
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 1)).unwrap() {
                        panic!("Piece at 1,1 after pawn moved")
                    }
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 2)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 2),
                            "Pawn is incorrect about location after move"
                        )
                    }
                }
                #[test]
                fn black_intended() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 6),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,6")
                    }
                    assert!(game.play(&Pos::new(1, 6), &Pos::new(1, 5)));
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 6)).unwrap() {
                        panic!("Piece at 1,6 after pawn moved")
                    }
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 5)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 5),
                            "Pawn is incorrect about location after move"
                        )
                    }
                }
                #[test]
                fn white_cannot_move_out_of_board() {
                    let mut game: Game = Game::new();
                    assert!(game
                        .add(
                            &Pos::new(0, 7),
                            &Piece::Pawn(Pawn {
                                pos: Pos::new(0, 7),
                                team: Team::White
                            })
                        )
                        .unwrap());
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(0, 7)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(0, 7),
                            "Pawn is incorrect about location on start"
                        );
                    }
                    assert!(!game.play(&Pos::new(0, 7), &Pos::new(0, 8)));
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(0, 7)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(0, 7),
                            "Pawn is incorrect about location on failed move"
                        );
                    }
                }
                #[test]
                fn white_cannot_double_move() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 1),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,1")
                    }
                    assert!(
                        !game.play(&Pos::new(1, 1), &Pos::new(1, 3)),
                        "Pawn double moved"
                    );
                }
                #[test]
                fn black_cannot_double_move() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 6),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,6")
                    }
                    assert!(
                        !game.play(&Pos::new(1, 6), &Pos::new(1, 4)),
                        "Pawn double moved"
                    );
                }
                #[test]
                fn white_cannot_attack() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 1),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,1")
                    }
                    assert!(!game
                        .add(
                            &Pos::new(1, 2),
                            &Piece::Pawn(Pawn {
                                pos: Pos::new(1, 2),
                                team: Team::Black
                            })
                        )
                        .unwrap());
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 2)).unwrap().unwrap()
                    {
                        assert_eq!(piece.pos, Pos::new(1, 2), "Pos was not added correctly");
                        assert_eq!(piece.team, Team::Black, "Team was not added correctly");
                    }
                    assert!(
                        !game.play(&Pos::new(1, 1), &Pos::new(1, 2)),
                        "Pawn attacked"
                    );
                }
                #[test]
                fn black_cannot_attack() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 6),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,6")
                    }
                    assert!(!game
                        .add(
                            &Pos::new(1, 5),
                            &Piece::Pawn(Pawn {
                                pos: Pos::new(1, 5),
                                team: Team::White
                            })
                        )
                        .unwrap());
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 5)).unwrap().unwrap()
                    {
                        assert_eq!(piece.pos, Pos::new(1, 5), "Pos was not added correctly");
                        assert_eq!(piece.team, Team::White, "Team was not added correctly");
                    }
                    assert!(
                        !game.play(&Pos::new(1, 6), &Pos::new(1, 5)),
                        "Pawn attacked"
                    );
                }
                #[test]
                fn white_cannot_do_friendly_fire() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 1),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,1")
                    }
                    assert!(!game
                        .add(
                            &Pos::new(1, 2),
                            &Piece::Pawn(Pawn {
                                pos: Pos::new(1, 2),
                                team: Team::White
                            })
                        )
                        .unwrap());
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 2)).unwrap().unwrap()
                    {
                        assert_eq!(piece.pos, Pos::new(1, 2), "Pos was not added correctly");
                        assert_eq!(piece.team, Team::White, "Team was not added correctly");
                    }
                    assert!(
                        !game.play(&Pos::new(1, 1), &Pos::new(1, 2)),
                        "Pawn did friendly fire"
                    );
                }
                #[test]
                fn black_cannot_do_friendly_fire() {
                    let mut game: Game = Game::new();
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        assert_eq!(
                            piece.pos,
                            Pos::new(1, 6),
                            "Pawn is incorrect about location on start"
                        )
                    } else {
                        panic!("No pawn at 1,6")
                    }
                    assert!(!game
                        .add(
                            &Pos::new(1, 5),
                            &Piece::Pawn(Pawn {
                                pos: Pos::new(1, 5),
                                team: Team::Black
                            })
                        )
                        .unwrap());
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 5)).unwrap().unwrap()
                    {
                        assert_eq!(piece.pos, Pos::new(1, 5), "Pos was not added correctly");
                        assert_eq!(piece.team, Team::Black, "Team was not added correctly");
                    }
                    assert!(
                        !game.play(&Pos::new(1, 6), &Pos::new(1, 5)),
                        "Pawn did friendly fire"
                    );
                }
            }
            mod left_diagonal {
                use super::super::super::super::*;
                #[test]
                fn white_intented() {
                    let mut game: Game = Game::new();
                    assert!(
                        !game
                            .add(
                                &Pos::new(0, 2),
                                &Piece::Pawn(Pawn {
                                    pos: Pos::new(0, 2),
                                    team: Team::Black
                                })
                            )
                            .unwrap(),
                        "Piece was at 0,2"
                    );
                    assert!(
                        game.play(&Pos::new(1, 1), &Pos::new(0, 2)),
                        "Could not take diagonally"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 1)).unwrap() {
                        panic!("Piece at previous location");
                    }
                    match get_pos(&game.board, &Pos::new(0, 2)).unwrap() {
                        Some(piece) => {
                            assert_eq!(
                                piece,
                                Piece::Pawn(Pawn {
                                    pos: Pos::new(0, 2),
                                    team: Team::White
                                })
                            )
                        }
                        None => {
                            panic!("No pice at end location")
                        }
                    }
                }
                #[test]
                fn black_intended() {
                    let mut game: Game = Game::new();
                    assert!(
                        !game
                            .add(
                                &Pos::new(0, 5),
                                &Piece::Pawn(Pawn {
                                    pos: Pos::new(0, 5),
                                    team: Team::White
                                })
                            )
                            .unwrap(),
                        "Piece was at 0,5"
                    );
                    assert!(
                        game.play(&Pos::new(1, 6), &Pos::new(0, 5)),
                        "Could not take diagonally"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 6)).unwrap() {
                        panic!("Piece at previous location");
                    }
                    match get_pos(&game.board, &Pos::new(0, 5)).unwrap() {
                        Some(piece) => {
                            assert_eq!(
                                piece,
                                Piece::Pawn(Pawn {
                                    pos: Pos::new(0, 5),
                                    team: Team::Black
                                })
                            )
                        }
                        None => {
                            panic!("No pice at end location")
                        }
                    }
                }
                #[test]
                fn white_cannot_move() {
                    let mut game: Game = Game::new();
                    let start: Pawn;
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        start = piece
                    } else {
                        panic!("1,1 weren't pawn")
                    };
                    assert_eq!(
                        start,
                        Pawn {
                            pos: Pos::new(1, 1),
                            team: Team::White
                        },
                        "Board generated incorrectly"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(0, 2)).unwrap() {
                        panic!("0,2 wasn't empty")
                    }
                    assert!(
                        !game.play(&start.pos, &Pos::new(0, 2)),
                        "Pawn moved diagonally"
                    );
                }
                #[test]
                fn black_cannot_move() {
                    let mut game: Game = Game::new();
                    let start: Pawn;
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        start = piece
                    } else {
                        panic!("1,6 weren't pawn")
                    };
                    assert_eq!(
                        start,
                        Pawn {
                            pos: Pos::new(1, 6),
                            team: Team::Black
                        },
                        "Board generated incorrectly"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(0, 5)).unwrap() {
                        panic!("0,5 wasn't empty")
                    }
                    assert!(
                        !game.play(&start.pos, &Pos::new(0, 5)),
                        "Pawn moved diagonally"
                    );
                }
            }
            mod right_diagonal {
                use super::super::super::super::*;
                #[test]
                fn white_intented() {
                    let mut game: Game = Game::new();
                    assert!(
                        !game
                            .add(
                                &Pos::new(2, 2),
                                &Piece::Pawn(Pawn {
                                    pos: Pos::new(2, 2),
                                    team: Team::Black
                                })
                            )
                            .unwrap(),
                        "Piece was at 2,2"
                    );
                    assert!(
                        game.play(&Pos::new(1, 1), &Pos::new(2, 2)),
                        "Could not take diagonally"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 1)).unwrap() {
                        panic!("Piece at previous location");
                    }
                    match get_pos(&game.board, &Pos::new(2, 2)).unwrap() {
                        Some(piece) => {
                            assert_eq!(
                                piece,
                                Piece::Pawn(Pawn {
                                    pos: Pos::new(2, 2),
                                    team: Team::White
                                })
                            )
                        }
                        None => {
                            panic!("No pice at end location")
                        }
                    }
                }
                #[test]
                fn black_intended() {
                    let mut game: Game = Game::new();
                    assert!(
                        !game
                            .add(
                                &Pos::new(2, 5),
                                &Piece::Pawn(Pawn {
                                    pos: Pos::new(2, 5),
                                    team: Team::White
                                })
                            )
                            .unwrap(),
                        "Piece was at 2,5"
                    );
                    assert!(
                        game.play(&Pos::new(1, 6), &Pos::new(2, 5)),
                        "Could not take diagonally"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(1, 6)).unwrap() {
                        panic!("Piece at previous location");
                    }
                    match get_pos(&game.board, &Pos::new(2, 5)).unwrap() {
                        Some(piece) => {
                            assert_eq!(
                                piece,
                                Piece::Pawn(Pawn {
                                    pos: Pos::new(2, 5),
                                    team: Team::Black
                                })
                            )
                        }
                        None => {
                            panic!("No pice at end location")
                        }
                    }
                }
                #[test]
                fn white_cannot_move() {
                    let mut game: Game = Game::new();
                    let start: Pawn;
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 1)).unwrap().unwrap()
                    {
                        start = piece
                    } else {
                        panic!("1,1 weren't pawn")
                    };
                    assert_eq!(
                        start,
                        Pawn {
                            pos: Pos::new(1, 1),
                            team: Team::White
                        },
                        "Board generated incorrectly"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(2, 2)).unwrap() {
                        panic!("2,2 wasn't empty")
                    }
                    assert!(
                        !game.play(&start.pos, &Pos::new(0, 2)),
                        "Pawn moved diagonally"
                    );
                }
                #[test]
                fn black_cannot_move() {
                    let mut game: Game = Game::new();
                    let start: Pawn;
                    if let Piece::Pawn(piece) =
                        get_pos(&game.board, &Pos::new(1, 6)).unwrap().unwrap()
                    {
                        start = piece
                    } else {
                        panic!("1,6 weren't pawn")
                    };
                    assert_eq!(
                        start,
                        Pawn {
                            pos: Pos::new(1, 6),
                            team: Team::Black
                        },
                        "Board generated incorrectly"
                    );
                    if let Some(_) = get_pos(&game.board, &Pos::new(2, 5)).unwrap() {
                        panic!("2,5 wasn't empty")
                    }
                    assert!(
                        !game.play(&start.pos, &Pos::new(2, 5)),
                        "Pawn moved diagonally"
                    );
                }
            }
        }
        #[allow(non_snake_case)]
        mod Knight {
            mod intended_move {
                use super::super::super::super::*;
                #[test]
                fn down_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn up_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn up_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn down_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_right() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_left() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap();
                }
            }
            mod white_intended_take {
                use super::super::super::super::*;
                #[test]
                fn down_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_right() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_left() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap();
                }
            }
            mod black_intended_take {
                use super::super::super::super::*;
                #[test]
                fn down_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_left_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 5),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_up_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 5),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 5),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn up_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 4),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_right_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 3),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_right() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
                #[test]
                fn down_down_left() {
                    play(
                        &Pos::new(3, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Knight(Knight {
                            pos: Pos::new(3, 4),
                            team: Team::Black,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap();
                }
            }
        }
        #[allow(non_snake_case)]
        mod Rook {
            mod intended_move {
                mod adjacent {
                    use super::super::super::super::super::*;
                    #[test]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(2, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 4),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(4, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn down() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                }
                mod far {
                    use super::super::super::super::super::*;
                    #[test]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(0, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 5),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(7, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                    #[test]
                    fn down() {
                        play(
                            &Pos::new(4, 4),
                            &Pos::new(4, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(4, 4),
                                team: Team::White,
                            })),
                            None,
                        )
                        .unwrap()
                    }
                }
            }
            mod intended_take {
                mod adjacent {
                    use super::super::super::super::super::*;
                    #[test]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(2, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(2, 3),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 4),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 4),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(4, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(4, 3),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn down() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 2),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                }
                mod far {
                    use super::super::super::super::super::*;
                    #[test]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(0, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(0, 3),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 5),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 5),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(7, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(7, 3),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    fn down() {
                        play(
                            &Pos::new(4, 4),
                            &Pos::new(4, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(4, 4),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(4, 2),
                                team: Team::Black,
                            })),
                        )
                        .unwrap()
                    }
                }
            }
            mod diagonal_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Rook(Rook {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Rook(Rook {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Rook(Rook {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Rook(Rook {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod take_friendly_fail {
                mod adjacent {
                    use super::super::super::super::super::*;
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(2, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(2, 3),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 4),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 4),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(4, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(4, 3),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn down() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 2),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                }
                mod far {
                    use super::super::super::super::super::*;
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn left() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(0, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(0, 3),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn up() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(3, 5),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(3, 5),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn right() {
                        play(
                            &Pos::new(3, 3),
                            &Pos::new(7, 3),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(3, 3),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(7, 3),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                    #[test]
                    #[should_panic(expected = "unwrap")]
                    fn down() {
                        play(
                            &Pos::new(4, 4),
                            &Pos::new(4, 2),
                            Some(Piece::Rook(Rook {
                                pos: Pos::new(4, 4),
                                team: Team::White,
                            })),
                            Some(Piece::Pawn(Pawn {
                                pos: Pos::new(4, 2),
                                team: Team::White,
                            })),
                        )
                        .unwrap()
                    }
                }
            }
        }
        #[allow(non_snake_case)]
        mod Bishop {
            mod adjacent_move {
                use super::super::super::super::*;
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod adjacent_take {
                use super::super::super::super::*;
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
            }
            mod far_move {
                use super::super::super::super::*;
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 5),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 5),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(6, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod far_take {
                use super::super::super::super::*;
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 5),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 5),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(6, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(6, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
            }
            mod take_friendly_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
            }
            mod horizonal_vertical_adjacent_move_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod horizonal_vertical_far_move_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(0, 3),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 5),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(7, 3),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Bishop(Bishop {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
        }
        #[allow(non_snake_case)]
        mod Queen {
            mod adjacent_move {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod adjacent_take {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
            }
            mod take_friendly_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 3),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 3),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
            }
            mod far_move {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(0, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(7, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(6, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod far_take {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(0, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(0, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(7, 3),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(7, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(4, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(1, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 5),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(5, 5),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(6, 2),
                        Some(Piece::Queen(Queen {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(6, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
            }
        }
        #[allow(non_snake_case)]
        mod King {
            mod adjacent_move {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
            mod adjacent_take {
                use super::super::super::super::*;
                #[test]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 3),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::Black,
                        })),
                    )
                    .unwrap()
                }
            }
            mod take_firendly_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 3),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 3),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(3, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(2, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        Some(Piece::Pawn(Pawn {
                            pos: Pos::new(4, 2),
                            team: Team::White,
                        })),
                    )
                    .unwrap()
                }
            }
            mod outside_range_fail {
                use super::super::super::super::*;
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_middle() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_up_corner() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(1, 5),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn top_left() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(2, 5),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn top_middle() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(3, 5),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn top_right() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(4, 5),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn up_right_corner() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 5),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_up() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 4),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_middle() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down() {
                    play(
                        &Pos::new(3, 3),
                        &Pos::new(5, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(3, 3),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn right_down_corner() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(6, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn bottom_right() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(5, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn bottom_middle() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(4, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn bottom_left() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(3, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn down_left_corner() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 2),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
                #[test]
                #[should_panic(expected = "unwrap")]
                fn left_down() {
                    play(
                        &Pos::new(4, 4),
                        &Pos::new(2, 3),
                        Some(Piece::King(King {
                            pos: Pos::new(4, 4),
                            team: Team::White,
                        })),
                        None,
                    )
                    .unwrap()
                }
            }
        }
    }
}
