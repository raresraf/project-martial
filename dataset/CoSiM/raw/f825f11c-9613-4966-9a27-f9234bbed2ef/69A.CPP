#include <iostream>
#include <string>

using namespace std;

int main() {

  int n;
  int x[100], y[100], z[100];
  cin >> n;
  int sumx = 0;
  int sumy = 0;
  ;
  int sumz = 0;
  int i = 0;

  while (n--) {
    cin >> x[i] >> y[i] >> z[i];
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
    i++;
  }

  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
