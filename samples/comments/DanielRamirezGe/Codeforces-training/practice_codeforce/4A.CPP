#include <bits/stdc++.h>
using namespace std;
int main() {
  int num;
  cin >> num;
  if ((num > 2) and ((num - 2) % 2) == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}