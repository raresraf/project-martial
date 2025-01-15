#include <iostream>
using namespace std;
int main() {
  string s;
  getline(cin, s);
  char hello[] = {'h', 'e', 'l', 'l', 'o'};
  int j = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == hello[j])
      j++;
  }
  if (j == 5)
    cout << "YES";
  else {
    cout << "NO";
  }
}