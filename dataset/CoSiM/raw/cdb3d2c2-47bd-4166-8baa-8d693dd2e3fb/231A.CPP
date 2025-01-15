
/*hat bei codeforces funktioniert kein plan warum hier nicht evtl shadow bytes
 * aber was ist das  berhaupt well fuck it  */

#include <iostream>

using namespace std;

int main() {
  int a, sum;
  cin >> a;
  int b[a][3];
  sum = 0;

  /*c = (sizeof(b)/sizeof(*b)); */

  for (int i = 0; i <= a; i++) {
    cin >> b[i][0] >> b[i][1] >> b[i][2];

    if (b[i][0] == 1 and b[i][1] == 1) {
      sum++;
    } else if (b[i][1] == 1 and b[i][2] == 1) {
      sum++;
    } else if (b[i][0] == 1 and b[i][2] == 1) {
      sum++;
    }
  }

  cout << sum << endl;
}
