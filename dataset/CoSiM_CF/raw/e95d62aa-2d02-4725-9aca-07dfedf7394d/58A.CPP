#include <iostream>
using namespace std;

int main() {
  string word;
  cin >> word;
  int n = word.length();
  string goal = "hello";
  int x = 0;
  string result = "NO";
  for (int i = 0; i < n; i++) {
    if (word[i] == goal[x])
      x++;
    if (x == 5)
      result = "YES";
  }
  cout << result;
  cout << "\n";
}
