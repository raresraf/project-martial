// Codeforces
// 266A - Stone on the tables

#include <iostream>
using namespace std;

int main() {
  int nbrStones;
  cin >> nbrStones;
  cin.get();

  string s;
  getline(cin, s);

  int stonesToTake = 0;

  for (unsigned int i = 0; i < s.size() - 1; i++) {
    if (s[i] == s[i + 1])
      stonesToTake += 1;
  }

  cout << stonesToTake;

  return 0;
}
