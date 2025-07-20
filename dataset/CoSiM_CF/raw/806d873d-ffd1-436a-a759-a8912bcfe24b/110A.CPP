#include <bits/stdc++.h>
using namespace std;
int main() {
  int check = 0;
  char num[100];
  cin >> num;
  int x = strlen(num);

  for (int i = 0; i < x; i++) {
    if (num[i] == '7' || num[i] == '4')
      check++;
  }

  if (check == 4 || check == 7)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
