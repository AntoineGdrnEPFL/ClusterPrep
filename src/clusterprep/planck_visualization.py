"""Alignment QA: python -m clusterprep.planck_visualization --help."""
import argparse
import json
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from .io import load_celestial_fits, resolve_radio_path
from .dataset import ClusterDataset
from .planck import project


def render(info, options, output, size=128, matched=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = resolve_radio_path(info)
    radio, _, wcs = load_celestial_fits(path)
    planck = project(options, wcs, radio.shape, str(Path(path).resolve()))
    xray = np.full(radio.shape, np.nan)
    if info.get('chandra_path'):
        image, _, xwcs = load_celestial_fits(info['chandra_path'])
        xray, footprint = reproject_interp((image,xwcs), wcs, shape_out=radio.shape)
        xray[footprint <= 0] = np.nan
    ds = ClusterDataset({'qa':info}, ['qa'], [], size=size, mode='multimodal',
        modalities=['radio','xray','planck'], planck=options, include_targets=False,
        processed_dir=None, radio_preprocessing='matched_resolution' if matched else 'historical')
    cropped = ds._modality_images(ds.data_info['qa'])
    center = wcs.pixel_to_world((radio.shape[1]-1)/2,(radio.shape[0]-1)/2)
    wide = WCS(naxis=2)
    wide.wcs.ctype=['RA---TAN','DEC--TAN']
    wide.wcs.crval=[center.icrs.ra.deg, center.icrs.dec.deg]
    wide.wcs.crpix=[128.5,128.5]
    extent=max(2.,max(np.array(radio.shape)*proj_plane_pixel_scales(wcs))*2)
    wide.wcs.cdelt=[-extent/256,extent/256]
    large=project(options,wide,(256,256),'qa-wide:'+info['cluster_id'])
    fig=plt.figure(figsize=(16,14),layout='constrained')
    def panel(position, image, grid, title):
        ax=fig.add_subplot(3,3,position,projection=grid)
        values=image[np.isfinite(image)]
        low,high=np.percentile(values,[2,98]) if values.size else (0,1)
        im=ax.imshow(image,origin='lower',vmin=low,vmax=high,cmap='viridis')
        ax.set_title(f'{title}\n{image.shape}; coverage={np.isfinite(image).mean():.1%}',fontsize=10)
        ax.coords[0].set_axislabel('RA');ax.coords[1].set_axislabel('Dec')
        fig.colorbar(im,ax=ax,shrink=.65)
        return ax
    for i,(name,image) in enumerate([('Radio',radio),('CHANDRA',xray),('Planck y',planck['planck'])],1):
        panel(i,image,wcs,name+' before crop')
    for i,(name,key) in enumerate([('Radio','raw'),('CHANDRA','chandra'),('Planck y','planck')],4):
        panel(i,cropped.get(key,np.full((size,size),np.nan)),cropped['wcs'],name+' after crop')
    for position,background,overlay,grid in [(7,radio,planck['planck'],wcs),
            (8,cropped['raw'],cropped['planck'],cropped['wcs'])]:
        ax=panel(position,background,grid,'Radio + Planck contours')
        values=overlay[np.isfinite(overlay)]
        if values.size and np.ptp(values)>0:
            levels=np.unique(np.percentile(values,[20,40,60,80]))
            ax.contour(overlay,levels=levels,colors='white',linewidths=.8)
    ax=panel(9,large['planck'],wide,'Wide Planck field; radio footprint in red')
    for grid,shape,color in [(wcs,radio.shape,'red'),(cropped['wcs'],(size,size),'cyan')]:
        h,w=shape
        xx=np.array([-.5,w-.5,w-.5,-.5,-.5]);yy=np.array([-.5,-.5,h-.5,h-.5,-.5])
        ra,dec=grid.pixel_to_world_values(xx,yy)
        ax.plot(ra,dec,color=color,transform=ax.get_transform('world'))
    arcsec=proj_plane_pixel_scales(cropped['wcs'])*3600
    fig.suptitle(f"{info['cluster_id']} | Planck cache: {planck['cache_status']} | "
                 f"crop pixel: {arcsec[0]:.2f} x {arcsec[1]:.2f} arcsec\n"
                 'Planck instrumental FWHM ~10 arcmin: finer pixels do not improve resolution',fontsize=13)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=130);plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radio',required=True)
    parser.add_argument('--chandra')
    parser.add_argument('--cluster',default='cluster')
    parser.add_argument('--source',default='planck_PR4_data/PR4_NILC_y_map.fits')
    parser.add_argument('--cache-dir',default='processed/planck_PR4')
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--output',required=True)
    parser.add_argument('--matched-info',help='JSON sample including radio_paths and physical_record; enables matched geometry')
    args=parser.parse_args()
    info=dict(cluster_id=args.cluster,raw_path=args.radio,chandra_path=args.chandra)
    if args.matched_info:
        info.update(json.loads(Path(args.matched_info).read_text()))
    print(render(info,dict(source_path=args.source,cache_dir=args.cache_dir),args.output,args.size,bool(args.matched_info)))


if __name__=='__main__':
    main()
