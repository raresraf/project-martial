#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

void convertLower(string &s) {
  for (int i = 0; i < s.size(); i++) {
    if (s[i] < 'a') {
      s[i] += 'a' - 'A';
    }
  }
}

int main() {
  string a, b;
  cin >> a >> b;

  convertLower(a);
  convertLower(b);

  int decision = 0;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] < b[i]) {
      decision = -1;
      break;
    } else if (a[i] > b[i]) {
      decision = 1;
      break;
    }
  }
  cout << decision;
  return 0;
}