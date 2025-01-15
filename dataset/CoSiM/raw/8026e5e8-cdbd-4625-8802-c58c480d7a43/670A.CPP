#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, mini, maxi;
  cin >> n;
  mini = (n / 7) * 2;
  if (n % 7 == 6)
    mini++;
  if (n % 7 >= 2)
    maxi = ((n / 7) * 2) + 2;
  else if (n % 7 == 1)
    maxi = ((n / 7) * 2) + 1;
  else
    maxi = ((n / 7) * 2);
  cout << mini << " " << maxi << endl;
  return 0;
}
