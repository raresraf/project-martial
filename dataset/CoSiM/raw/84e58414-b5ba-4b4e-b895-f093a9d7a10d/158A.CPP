#include <iostream>
using namespace std;
int main() {
  int n, k, kscore, score, i;
  cin >> n >> k;
  for (i = 0; i < n; i++) {
    cin >> score;
    if (score == 0)
      break; // demand positive score
    if (i == k - 1)
      kscore = score;
    if (i > k - 1 && score < kscore)
      break;
  }
  cout << i;
  return 0;
}
// wrong input: 4 2 \n 0 0 0 0
// wrong reason: demand positive score in the problem
/**wrong code:
if(n==k)cout<<k;
    else{
        for (i=0;i<n;++i){
        cin>>score;
        if(i==k-1)kscore = score;
        if(i>k-1&&score<kscore)break;
        }
        cout<<i;
    }
**/
