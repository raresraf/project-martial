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
  bool dangerous = false;
  int streak = 0;
  char prev = '2';
  for (char c : s) {
    if (c == prev) {
      streak++;
    } else {
      streak = 1;
    }
    prev = c;
    if (streak == 7) {
      dangerous = true;
      break;
    }
  }

  cout << (dangerous ? "YES" : "NO");
  return 0;
}