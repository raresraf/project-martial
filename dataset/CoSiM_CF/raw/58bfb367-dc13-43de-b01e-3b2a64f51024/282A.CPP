#include <iostream>
#include <string>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  lli n;
  cin >> n;
  lli x = 0;
  while (n--) {
    string a;
    cin >> a;
    if (isalpha(a[0]) && (a[1]) == '+' && a[2] == '+') {
      x++;
    } else if (isalpha(a[2]) && (a[0] == '+') && (a[1] == '+')) {
      x++;
    } else if (isalpha(a[2]) && (a[0] == '-') && (a[1] == '-'))
      x--;
    else if (isalpha(a[0]) && (a[1] == '-') && (a[2] == '-'))
      x--;
  }
  cout << x << endl;
  return 0;
}