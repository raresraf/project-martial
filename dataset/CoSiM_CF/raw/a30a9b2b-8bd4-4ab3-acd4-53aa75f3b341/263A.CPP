/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h>

using namespace std;

#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

#define D(x) cout << #x << " = " << x << endl;

int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  int mat[5][5], res;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      scanf("%d", &mat[i][j]);
      if (mat[i][j] == 1) {
        res = (abs(i - 2) + abs(j - 2));
      }
    }
  printf("%d\n", res);
  return 0;
}