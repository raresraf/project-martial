// by mo1ein
#include <iostream>
using namespace std;
int main() {
  int n = 0, problems[1000][3], count_line = 0, count = 0;
  cin >> n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++)
      cin >> problems[i][j];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      if (problems[i][j] == 1)
        count_line++; // count every line
    }
    if (count_line >= 2)
      count++; // count if can solve ? in line
    count_line = 0;
  }
  cout << count;
  return 0;
}
