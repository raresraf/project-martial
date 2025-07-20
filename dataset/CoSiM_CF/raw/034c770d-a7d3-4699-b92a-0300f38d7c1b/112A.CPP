#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int i, n = 0;
int sum;
string ch;
int main() {
  cin >> ch;
  sort(ch.begin(), ch.end());
  for (i = 0; i < ch.length(); i++) {
    if (ch[i] == ch[i + 1]) {
      continue;
    } else {
      n++;
    }
  }
  if (n % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
