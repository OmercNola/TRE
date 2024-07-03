set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'NLKNguyen/papercolor-theme'
" Plugin 'valloric/youcompleteme'
" Plugin 'tpope/vim-commentary'

call vundle#end()            " required
filetype plugin indent on    " required

" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal

set number
syntax on
set t_Co=256
set background=dark
colorscheme PaperColor
filetype plugin indent on
set autoindent 
set tabstop=4
set shiftwidth=4
set expandtab
inoremap kj <Esc>
set hlsearch
