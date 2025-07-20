#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  string s;
  cin >> s;

  int to_remove = 0;
  char prev = s[0];

  for (int i = 1; i < n; i++) {
    while (prev == s[i]) {
      to_remove++;
      i++;
    }

    prev = s[i];
  }

  cout << to_remove << endl;

  return 0;
}
