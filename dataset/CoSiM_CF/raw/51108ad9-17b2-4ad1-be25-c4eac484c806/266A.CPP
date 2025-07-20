#include <iostream>
using namespace std;

int main() {

  int color = 0;
  int result = 0;
  char c;
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> c;
    if ((c == 'R') && (color == 1))
      result++;
    if ((c == 'G') && (color == 2))
      result++;
    if ((c == 'B') && (color == 3))
      result++;
    if (c == 'R')
      color = 1;
    if (c == 'G')
      color = 2;
    if (c == 'B')
      color = 3;
  }
  cout << result;
  cout << "\n";
}
