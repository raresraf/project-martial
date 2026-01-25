#include <bits/stdc++.h>
// typedef vector<int> vint;
#define pb push_back
using namespace std;
/*int main()
{
  string s,ss;
  cin>>s>>ss;
  vector<int> vec;
  int l1=s.length();
  int l2=ss.length();
  if(l1!=l2)cout<<"NO"<<endl;
  else{
       for(int i=0;i<l1;i++){
       if(s[i]!=ss[i])vec.pb(i);

   }

//for(int i=0;i<vec.size();i++)cout<<vec[i]<<" ";
   int l3=vec.size();
   if(l3==2)
   {
       if(s[vec[0]]==ss[vec[1]]  && s[vec[1]]==ss[vec[0]])
        cout<<"YES"<<endl;
       else cout<<"NO"<<endl;
   }
   else cout<<"NO"<<endl;
  }
  return 0;
}*/
int main() {
  string s, ss;
  int count = 0;
  cin >> s >> ss;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != ss[i])
      count++;
  }
  sort(s.begin(), s.end());
  sort(ss.begin(), ss.end());
  if (count == 2 && s == ss)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
