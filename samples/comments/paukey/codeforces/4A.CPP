#include <iostream>
using namespace std;
int w;

int main() {
  cin >> w;
  /*
  if(w>=4&&w%2==0){
          cout<<"YES"<<endl;
          }
  else{
          cout<<"NO"<<endl;
          }
  */
  // new
  if (w & 1 || w == 2)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}
