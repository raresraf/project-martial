#include <algorithm>
#include <cmath>
#include <complex>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define REV(i, b, a) for (int i = (a); i >= (b); i--)
#define mp make_pair
#define pb push_back

int main(int argc, char *argv[]) {

  unsigned short n;
  signed short X = 0;
  string s;
  vector<string> input;
  cin >> n;
  FOR(i, 0, n) {
    cin >> s;
    input.pb(s);
  }

  FOR(i, 0, n) {
    if (input.at(i).find("++") != std::string::npos) {
      X++;
    } else if (input.at(i).find("--") != std::string::npos) {
      X--;
    }
  }

  cout << X;

  return 0;
}
