#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  int t, n, v;
  scanf("%d", &t);
  while (t--) {
    scanf("%d", &n);
    ll maxNeg = 0, maxPos = 0;
    ll negLen = 0, posLen = 0;
    for (int i = 0; i < n; i++) {
      scanf("%d", &v);
      if (v > 0) {
        ll nextLen = negLen + 1;
        ll nextVal = maxNeg + v;
        if (nextLen > posLen) {
          posLen = nextLen;
          maxPos = nextVal;
        } else if (nextVal > maxPos)
          maxPos = nextVal;
      } else {
        ll nextLen = posLen + 1;
        ll nextVal = maxPos + v;
        if (nextLen > negLen) {
          negLen = nextLen;
          maxNeg = nextVal;
        } else if (nextVal > maxNeg)
          maxNeg = nextVal;
      }
    }
    ll bestLen = negLen;
    ll bestVal = maxNeg;
    if (posLen > bestLen || (posLen == bestLen && maxPos > bestVal)) {
      bestLen = posLen;
      bestVal = maxPos;
    }
    printf("%lld\n", bestVal);
  }
}