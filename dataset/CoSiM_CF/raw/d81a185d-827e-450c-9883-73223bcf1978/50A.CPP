#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;
int main() {
  int M, N, dominoSQUARE = 2;
  cin >> M >> N;
  if (1 <= M && M <= N && M <= 16 && N <= 16) {
    int board_square = M * N;
    int numberOFdominos = board_square / dominoSQUARE;
    cout << numberOFdominos;
  }
  return 0;
}
