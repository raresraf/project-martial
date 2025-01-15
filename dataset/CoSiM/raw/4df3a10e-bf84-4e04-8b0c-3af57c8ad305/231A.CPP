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
  int n, Petya, Vasya, Tonya, cunt = 0;
  cin >> n;
  while (n--) {
    cin >> Petya >> Vasya >> Tonya;
    if (Petya == 1 && Vasya == 1 || Petya == 1 && Tonya == 1 ||
        Vasya == 1 && Tonya == 1) {
      cunt++;
    }
  }
  cout << cunt;

  return 0;
}
