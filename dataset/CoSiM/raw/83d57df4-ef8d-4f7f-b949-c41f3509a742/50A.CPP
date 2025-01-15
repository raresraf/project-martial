#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli a, b;
  cin >> a >> b;
  cout << ((a * b) >> 1) << endl;
  return 0;
}
