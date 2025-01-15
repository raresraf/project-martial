#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  lli n;
  cin >> n;
  lli arr[n][3], solved = 0;
  for (int i = 0; i < n; ++i) {
    lli count = 0;
    for (int j = 0; j < 3; ++j) {
      cin >> arr[i][j];
      if (arr[i][j] == 1)
        count++;
    }
    if (count >= 2) {
      solved++;
    }
  }
  cout << solved << endl;
  return 0;
}