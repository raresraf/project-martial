#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ull unsigned long long
#define ll long long

using namespace std;

int main() {
  string s;
  cin >> s;
  int step = 0;
  for (char c : s) {
    if (step == 0 && c == 'h' || step == 1 && c == 'e' ||
        step == 2 && c == 'l' || step == 3 && c == 'l' ||
        step == 4 && c == 'o') {
      step++;
    }
  }

  cout << (step == 5 ? "YES" : "NO");
  return 0;
}