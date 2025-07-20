#include <bits/stdc++.h>
using namespace std;

int main() {
  string inp;
  cin >> inp;
  vector<char> s;

  char hello[5] = {'h', 'e', 'l', 'l', 'o'};
  bool found = false;
  int next = 0;

  int len = inp.length();
  for (int i = 0; i < len; i++) {
    if (inp[i] == hello[next])
      next++;

    if (next == 5) {
      found = true;
      break;
    }

    // cout << endl << next;
  }

  if (found)
    cout << "YES";
  else
    cout << "NO";
}
