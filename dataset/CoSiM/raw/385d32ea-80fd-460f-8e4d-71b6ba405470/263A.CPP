#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

using namespace std;

int arr[5][5];

int main(int argc, char const *argv[]) {
  int x, y;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  int test = abs(3 - x) + abs(3 - y);
  cout << test;
  return 0;
}