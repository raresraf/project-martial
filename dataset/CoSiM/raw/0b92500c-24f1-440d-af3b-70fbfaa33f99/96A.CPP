// 96A - Football

#include <iostream>

using namespace std;

int main() {
  string team;
  int i, sum0 = 0, sum1 = 0;
  bool danger = false;

  cin >> team;

  for (i = 0; i < team.size(); i++) {
    if ((team[i]) == '0') {
      sum0++;
      sum1 = 0;

      if (sum0 >= 7) {
        danger = true;
        break;
      }
    } else {
      sum1++;
      sum0 = 0;

      if (sum1 >= 7) {
        danger = true;
        break;
      }
    }
  }

  if (danger)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
