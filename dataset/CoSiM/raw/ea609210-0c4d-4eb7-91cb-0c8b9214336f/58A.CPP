#include <iostream>
using namespace std;
int main() {
  string s;
  string goal("hello");
  int index = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == goal[index])
      index++;
    if (index == 5) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}
