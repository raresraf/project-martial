// 110A - Nearly Lucky Number

#include <iostream>

using namespace std;

int main() {
  string n;
  int i, sum = 0;

  cin >> n;

  for (i = 0; i < n.size(); i++) {
    if (n[i] == '4' || n[i] == '7')
      sum++;
  }

  if (sum == 4 || sum == 7)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
