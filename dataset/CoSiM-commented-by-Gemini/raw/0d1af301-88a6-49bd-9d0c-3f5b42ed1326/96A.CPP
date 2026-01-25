#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

string solve(string s) {
  int counter = 0;

  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] == s[i + 1]) {
      counter++;

      if (counter == 6) {
        return "YES";
      }

      continue;
    }
    counter = 0;
  }
  return "NO";
}

void test() {
  assert(solve("1000000001") == "YES");
  assert(solve("00100110111111101 ") == "YES");
  assert(solve("001001") == "NO");
  assert(solve("11110111011101 ") == "NO");
}

int main() {
  // freopen( "input.txt", "r", stdin );

  string s;

  cin >> s;

  // test();

  cout << solve(s) << endl;
}
